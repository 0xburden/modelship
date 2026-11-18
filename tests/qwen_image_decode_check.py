"""Weight-free numerical stitcher checks using NumPy as the tensor stand-in.

CUDA and the actual pinned Diffusers VAE are checked separately by the
same-latent diagnostic, not simulated by these tests.
"""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "internal/modelship/assets"))
from qwen_image_decode import HALO, SCALE, TILE, STRIDE, axis_tiles, install_tiled_decode


class Tensor(np.ndarray):
    device = "cpu-fixture"

    def clamp_(self, low, high):
        np.clip(self, low, high, out=self)
        return self

    def square(self):
        return np.square(self)

    def addcmul_(self, a, b):
        self += a * b
        return self


def torch_factory(factory):
    def create(arg, *, dtype, device):
        assert device == Tensor.device
        return factory(arg, dtype=dtype).view(Tensor)
    return create


class VAE:
    dtype = np.dtype("float32")
    config = SimpleNamespace(patch_size=None, z_dim=64, out_channels=4)
    use_tiling = True
    spatial_compression_ratio = SCALE
    tile_sample_min_height = tile_sample_min_width = TILE
    tile_sample_stride_height = tile_sample_stride_width = STRIDE
    decode, encode, blend_v, blend_h = object(), object(), object(), object()

    def __init__(self, height, width):
        self.height, self.width = height, width
        self.calls = []
        self.cache_resets = 0
        self.bad_shape = False
        self.fail = False

    def clear_cache(self):
        self.cache_resets += 1
        self._feat_map = []

    def post_quant_conv(self, tile):
        return tile

    def decoder(self, tile, *, feat_cache, feat_idx, first_chunk):
        self.assert_cache(feat_cache, feat_idx, first_chunk)
        y, x = (int(tile[0, i, 0, 0, 0]) * SCALE for i in (0, 1))
        h, w = (n * SCALE for n in tile.shape[-2:])
        self.calls.append((y, x, h, w))
        if self.fail:
            raise ValueError("fixture decoder failed")
        out = expected(y, x, h, w)
        # Only internal tile borders are unreliable, including at corners.
        # Any contribution from these borders would corrupt the global ramp.
        if x:
            out[..., :HALO] = -5
        if x + w < self.width:
            out[..., -HALO:] = -5
        if y:
            out[..., :HALO, :] = -5
        if y + h < self.height:
            out[..., -HALO:, :] = -5
        return out[..., :-1] if self.bad_shape else out

    def assert_cache(self, feat_cache, feat_idx, first_chunk):
        assert feat_cache is self._feat_map and not feat_cache and feat_idx == [0] and first_chunk
        feat_cache.append("fixture cache")


def expected(y, x, h, w):
    ramp = (np.arange(y, y + h)[:, None] + 2 * np.arange(x, x + w)[None, :]) / 4096
    out = np.broadcast_to(ramp.astype("float32"), (1, 4, 1, h, w)).copy().view(Tensor)
    out[:, 3] = 1.5  # Clamp after stitching.
    return out


def latents(height, width):
    z = np.zeros((1, 64, 1, height // SCALE, width // SCALE), dtype="float32").view(Tensor)
    z[0, 0, 0] = np.arange(height // SCALE)[:, None]
    z[0, 1, 0] = np.arange(width // SCALE)[None, :]
    return z


class DecoderChecks(unittest.TestCase):
    def setUp(self):
        torch = ModuleType("torch")
        torch.float32 = np.dtype("float32")
        for name in ("zeros", "arange", "ones"):
            setattr(torch, name, torch_factory(getattr(np, name)))
        self.diffusers = ModuleType("diffusers")
        self.diffusers.__version__ = "0.42.1"
        self.diffusers.AutoencoderKLQwenImage21 = VAE
        self.enterContext(patch.dict(sys.modules, {
            "torch": torch, "diffusers": self.diffusers,
            "diffusers.models.autoencoders.vae": SimpleNamespace(DecoderOutput=SimpleNamespace),
        }))

    def test_lattice_full_coverage_and_overlap_for_every_latent_axis(self):
        for size in range(16, 1025, 16):
            tiles = axis_tiles(size)
            self.assertEqual(sum(tiles[-1]), size)
            seen = np.zeros(size, dtype=int)
            for start, length in tiles:
                self.assertLessEqual(length, TILE)
                seen[start:start + length] += 1
            self.assertTrue(((seen >= 1) & (seen <= 2)).all())
            for left, right in zip(tiles, tiles[1:]):
                self.assertEqual(left[0] + left[1] - right[0], TILE - STRIDE)

    def test_reconstruction_both_axes_short_tails_corners_and_boundaries(self):
        for h, w in ((256, 544), (544, 256), (272, 400), (512, 768), (1024, 1024)):
            with self.subTest(h=h, w=w):
                vae = VAE(h, w)
                install_tiled_decode(vae)
                self.assertIs(vae.encode, VAE.encode)
                self.assertIs(vae.decode, VAE.decode)
                self.assertIs(vae.blend_h, VAE.blend_h)
                self.assertIs(vae.blend_v, VAE.blend_v)
                out = vae.tiled_decode(latents(h, w)).sample
                np.testing.assert_allclose(out, expected(0, 0, h, w).clamp_(-1, 1), rtol=0, atol=2e-7)
                self.assertEqual(vae.calls, [(y,x,hh,ww) for y,hh in axis_tiles(h) for x,ww in axis_tiles(w)])
                self.assertEqual(vae.cache_resets, len(vae.calls) + 1)
                self.assertEqual(vae._feat_map, [])
                self.assertIsInstance(vae.tiled_decode(latents(h, w), return_dict=False), tuple)

    def test_unexpected_version_geometry_dtype_or_layout_fails_closed(self):
        for fields in ({"dtype": np.dtype("float16")}, {"use_tiling": False}, {"tile_sample_stride_width": 160},
                       {"config": SimpleNamespace(patch_size=2, z_dim=64, out_channels=4)}):
            vae = VAE(512, 512)
            vae.__dict__.update(fields)
            with self.assertRaisesRegex(RuntimeError, "reviewed FP32"):
                install_tiled_decode(vae)
            self.assertFalse(hasattr(vae, "tiled_decode"))
        vae = VAE(512, 512)
        with patch.object(self.diffusers, "__version__", "0.42.2"), self.assertRaisesRegex(RuntimeError, "reviewed FP32"):
            install_tiled_decode(vae)
        install_tiled_decode(vae)
        for invalid in (latents(512, 512)[:, :, 0], latents(512, 512).astype("float16"), latents(1040, 512)):
            with self.assertRaisesRegex(RuntimeError, "decoder input"):
                vae.tiled_decode(invalid)
        self.assertEqual(vae.calls, [])
        for field, error in (("bad_shape", RuntimeError), ("fail", ValueError)):
            with patch.object(vae, field, True), self.assertRaises(error):
                vae.tiled_decode(latents(512, 512))
            self.assertEqual(vae._feat_map, [])


if __name__ == "__main__":
    unittest.main()
