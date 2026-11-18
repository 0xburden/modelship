"""Bounded tiled output decode for the reviewed Qwen Image 2.1 FP32 VAE.

Only the instance's tiled decode is replaced. Its untiled decode, batch slicing,
conditioning encode, and upstream blend functions remain unchanged.
"""
from types import MethodType


TILE = 256
STRIDE = 192
SCALE = 16
OVERLAP = TILE - STRIDE
HALO = 24


def axis_tiles(size):
    """(start, length) in output pixels. Stop once a tile reaches the boundary."""
    tiles = []
    for start in range(0, size, STRIDE):
        length = min(TILE, size - start)
        tiles.append((start, length))
        if start + length == size:
            break
    return tiles


def _tiled_decode(self, z, return_dict=True):
    import torch
    from diffusers.models.autoencoders.vae import DecoderOutput

    if (z.ndim != 5 or z.shape[:3] != (1, 64, 1) or z.dtype != torch.float32
            or not all(1 <= n <= 64 for n in z.shape[-2:])):
        raise RuntimeError("unexpected Qwen Image 2.1 tiled decoder input")
    height, width = (n * SCALE for n in z.shape[-2:])
    rows, cols = axis_tiles(height), axis_tiles(width)
    # Discard each tile's unreliable 24px interior border in the 64px overlap;
    # use complementary smoothstep weights across the remaining 16px. There
    # are at most two tiles along an axis at any coordinate, so the separable
    # weights sum to one even at corners, with no per-pixel denominator needed.
    t = ((torch.arange(OVERLAP, dtype=z.dtype, device=z.device) + 0.5 - HALO)
         / (OVERLAP - 2 * HALO)).clamp_(0, 1)
    entering = t.square() * (3 - 2 * t)

    def windows(tiles):
        for i, (start, length) in enumerate(tiles):
            weights = torch.ones(length, dtype=z.dtype, device=z.device)
            if i:
                weights[:OVERLAP] = entering
            if i + 1 < len(tiles):
                weights[-OVERLAP:] = 1 - entering
            yield start, length, weights

    out = torch.zeros((1, 4, 1, height, width), dtype=z.dtype, device=z.device)
    columns = list(windows(cols))
    try:
        for y, h, wy in windows(rows):
            for x, w, wx in columns:
                self.clear_cache()
                self._conv_idx = [0]
                tile = z[..., y // SCALE:(y + h) // SCALE, x // SCALE:(x + w) // SCALE]
                decoded = self.decoder(self.post_quant_conv(tile), feat_cache=self._feat_map,
                                       feat_idx=self._conv_idx, first_chunk=True)
                if decoded.shape != (1, 4, 1, h, w) or decoded.dtype != z.dtype or decoded.device != z.device:
                    raise RuntimeError("unexpected Qwen Image 2.1 decoded tile")
                out[..., y:y + h, x:x + w].addcmul_(decoded, wy[:, None] * wx)
                del decoded
    finally:
        self.clear_cache()
    out.clamp_(-1, 1)
    return DecoderOutput(sample=out) if return_dict else (out,)


def install_tiled_decode(vae):
    import diffusers
    import torch
    from diffusers import AutoencoderKLQwenImage21

    if (diffusers.__version__ != "0.42.1" or type(vae) is not AutoencoderKLQwenImage21
            or vae.dtype != torch.float32 or not vae.use_tiling
            or vae.config.patch_size is not None or vae.config.z_dim != 64 or vae.config.out_channels != 4
            or (vae.spatial_compression_ratio, vae.tile_sample_min_height, vae.tile_sample_min_width,
                vae.tile_sample_stride_height, vae.tile_sample_stride_width) != (SCALE, TILE, TILE, STRIDE, STRIDE)):
        raise RuntimeError("Qwen Image 2.1 tiled decoder does not match the reviewed FP32 layout")
    vae.tiled_decode = MethodType(_tiled_decode, vae)
