#!/bin/bash
# Quick script to create the 3000-slice dataset subset

echo "🚀 Creating 3000-slice dataset subset..."
echo "This will take a few minutes..."

cd /home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR

python3 create_dataset_subset.py \
    --source "/home/Drive-D/UWSpine_adn" \
    --size 3000 \
    --seed 42

echo ""
echo "✅ Dataset subset creation complete!"
echo "🎯 Ready to run fine-tuning with:"
echo "   python main_train_scunet_ngswin_1.py --opt options/train_scunet_ngswin_finetune_synthetic_transfer.json"
