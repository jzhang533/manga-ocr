from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
)

class MyProcessor:
    def __init__(self, image_processor, tokenizer):
        self.feature_extractor = image_processor
        self.tokenizer = tokenizer


def get_processor(encoder_name, decoder_name):
    image_processor = AutoImageProcessor.from_pretrained(encoder_name, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = MyProcessor(image_processor, tokenizer)
    return processor

def get_model(encoder_name, decoder_name, max_length, num_decoder_layers=12):
    
    processor = get_processor(encoder_name, decoder_name)

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_name, 
        decoder_name, 
        decoder_num_hidden_layers=num_decoder_layers,
        )
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = max_length
    model.generation_config.early_stopping = True
    model.generation_config.num_beams = 4
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0

    return model, processor

if __name__ == "__main__":

    encoder_name="facebook/deit-tiny-patch16-224"
    decoder_name="tohoku-nlp/bert-base-japanese-char-v2"
    num_decoder_layers=2
    max_len=300

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)
    print(model)