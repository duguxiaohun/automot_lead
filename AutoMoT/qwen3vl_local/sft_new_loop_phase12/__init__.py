"""One freshly trained LoRA, two independent questions sharing an image prefix."""
CONTRACT = 'phase12_shared_lora_v5_output_parity_20261030'
MODES = ('2rgb_endpoints', '4rgb')
EPOCHS = 6
QUOTA = 1024
WORLD = 4
UPDATES = EPOCHS * QUOTA * 7 // WORLD
