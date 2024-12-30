
# evaluating on manga109s my split evaluation dataset (with 11361 items)
# /home/aistudio/mydata/manga/out/checkpoint-140000"
# cer = 0.10777018633540372, accuracy = 0.7010309278350515
# kha-white/manga-ocr-base
# cer = 0.10563975155279504, accuracy = 0.6701030927835051

from manga_ocr_dev.training.dataset import MangaDataset
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForVision2Seq
import torch
from manga_ocr_dev.training.my_get_model import get_processor
import numpy as np
from tqdm import tqdm
import pandas as pd

pretrained_model_name_or_path="kha-white/manga-ocr-base"
#pretrained_model_name_or_path="/home/aistudio/mydata/manga/out/checkpoint-140000"
use_cpu = False

if False:
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            print(f"setting logging level to error for f{logger.name}")
            logger.setLevel(logging.ERROR)


encoder_name="facebook/deit-tiny-patch16-224"
decoder_name="tohoku-nlp/bert-base-japanese-char-v2"
processor = get_processor(encoder_name, decoder_name)

tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-char-v2")
model = AutoModelForVision2Seq.from_pretrained(pretrained_model_name_or_path)
eval_dataset = MangaDataset(processor, "test", 300, augment=False, skip_packages=range(0, 9999))
if use_cpu:
    model.to('cpu')
else:
    model.to('cuda')
model.eval()

dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
cer_metric = evaluate.load("cer")

all_pred_str=np.array([])
all_label_str=np.array([])

for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
    x = batch['pixel_values']
    y = batch['labels']
    
    with torch.no_grad():
        if use_cpu:
            pred = model.generate(x.cpu(), max_length=300).cpu()
        else:
            pred = model.generate(x.cuda(), max_length=300).cpu()
        #print(pred.shape)
        pred_str = tokenizer.batch_decode(pred, skip_special_tokens=True)

        y[y == -100] = processor.tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(y, skip_special_tokens=True)

        pred_str = np.array(["".join(text.split()) for text in pred_str])
        label_str = np.array(["".join(text.split()) for text in label_str])
        all_pred_str = np.append(all_pred_str, pred_str)
        all_label_str = np.append(all_label_str, label_str)

cer = cer_metric.compute(predictions=all_pred_str, references=all_label_str)
accuracy = (pred_str == label_str).mean()

print(f"cer = {cer}, accuracy = {accuracy}")


df = pd.DataFrame({'pred': all_pred_str, 'label': all_label_str})

# Save the DataFrame to a CSV file
df.to_csv('pred_label_compare.csv', index=False)
