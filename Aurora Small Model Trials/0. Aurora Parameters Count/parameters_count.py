from aurora import AuroraPretrained

model = AuroraPretrained()
model.load_checkpoint()

# Assuming 'model' is your loaded pretrained Aurora model
total_params = sum(p.numel() for p in model.parameters())

# To count only the trainable parameters (recommended for fine-tuning scenarios)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters in Aurora Pretrained: {total_params}")
print(f"Trainable Parameters in Aurora Pretrained: {trainable_params}")


from aurora import AuroraSmallPretrained

model1 = AuroraSmallPretrained()
model1.load_checkpoint()

# Assuming 'model1' is your loaded pretrained Aurora Small model
total_params1 = sum(p.numel() for p in model1.parameters()) 

# To count only the trainable parameters (recommended for fine-tuning scenarios)
trainable_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)          

print(f"Total Parameters in Aurora Small Pretrained: {total_params1}")
print(f"Trainable Parameters in Aurora Small Pretrained: {trainable_params1}")

from aurora import Aurora

model2 = Aurora()
model2.load_checkpoint()

# Assuming 'model2' is your loaded Aurora Finetuned model
total_params2 = sum(p.numel() for p in model2.parameters())
# To count only the trainable parameters (recommended for fine-tuning scenarios)
trainable_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)

print(f"Total Parameters in Aurora Finetuned: {total_params2}")
print(f"Trainable Parameters in Aurora Finetuned: {trainable_params2}")

from aurora import Aurora12hPretrained

model3 = Aurora12hPretrained()
model3.load_checkpoint()

# Assuming 'model3' is your loaded pretrained Aurora 12h model
total_params3 = sum(p.numel() for p in model3.parameters())
# To count only the trainable parameters (recommended for fine-tuning scenarios)
trainable_params3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)

print(f"Total Parameters in Aurora 12h Pretrained: {total_params3}")
print(f"Trainable Parameters in Aurora 12h Pretrained: {trainable_params3}")

from aurora import AuroraHighRes

model4 = AuroraHighRes()
model4.load_checkpoint()

# Assuming 'model4' is your loaded Aurora HighRes model
total_params4 = sum(p.numel() for p in model4.parameters())
# To count only the trainable parameters (recommended for fine-tuning scenarios)
trainable_params4 = sum(p.numel() for p in model4.parameters() if p.requires_grad)

print(f"Total Parameters in Aurora HighRes 1 degree: {total_params4}")
print(f"Trainable Parameters in Aurora HighRes 1 degree: {trainable_params4}")

from aurora import AuroraWave

model5 = AuroraWave()
model5.load_checkpoint()

# Assuming 'model5' is your loaded Aurora Wave model
total_params5 = sum(p.numel() for p in model5.parameters())
# To count only the trainable parameters (recommended for fine-tuning scenarios)
trainable_params5 = sum(p.numel() for p in model5.parameters() if p.requires_grad)
print(f"Total Parameters in Aurora Wave: {total_params5}")
print(f"Trainable Parameters in Aurora Wave: {trainable_params5}")


