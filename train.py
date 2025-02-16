import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        frames, masks, labels = batch
        pred_masks, pred_labels = self(frames)
        
        # Multi-task loss
        seg_loss = DiceLoss()(pred_masks, masks)
        cls_loss = FocalLoss()(pred_labels, labels)
        total_loss = 0.7*seg_loss + 0.3*cls_loss
        
        self.log('train_loss', total_loss)
        return total_loss

# Hyperparameters
trainer = pl.Trainer(
    accelerator='gpu',
    precision=16,
    max_epochs=50
)
