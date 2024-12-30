import fire
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

from manga_ocr_dev.env import TRAIN_ROOT
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.my_get_model import get_model
from manga_ocr_dev.training.metrics import Metrics

if False:
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            print(f"setting logging level to error for f{logger.name}")
            logger.setLevel(logging.ERROR)

def run(
    run_name="debug",
    encoder_name="facebook/deit-tiny-patch16-224",
    decoder_name="tohoku-nlp/bert-base-japanese-char-v2",
    #decoder_name="rinna/japanese-gpt2-small",
    #max_len=300,   # whether set max_len to 128 is enough ? 
    max_len=300,
    num_decoder_layers=2,
    batch_size=128,
    num_epochs=500,
    fp16=True,
):

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)
    #data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model, padding=True)
    #print(model)
    train_dataset = MangaDataset(processor, "train", max_len, augment=True, skip_packages=[])
    #train_dataset = MangaDataset(processor, "train", max_len, augment=True, skip_packages=range(0, 99))
    eval_dataset = MangaDataset(processor, "test", max_len, augment=False, skip_packages=range(0, 9999))

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        fp16_full_eval=fp16,
        dataloader_num_workers=16,
        output_dir=TRAIN_ROOT,
        logging_steps=100,
        #report_to="wandb",
        report_to="none",
        save_steps=10000,
        eval_steps=10000,
        num_train_epochs=num_epochs,
        run_name=run_name,
        #resume_from_checkpoint= TRAIN_ROOT / 'checkpoint-30000'
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.feature_extractor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train(
        resume_from_checkpoint= True
    )


if __name__ == "__main__":
    fire.Fire(run)
