#pragma once

// Configure the library's Qwen BPE loader from the already verified encoder,
// before the private worker creates its sole model context.
bool modelship_load_qwen_vocab(const char *encoder_path);
