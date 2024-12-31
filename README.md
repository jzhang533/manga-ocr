
- My Training Logs
  - 2024-12-31, after several experiments, by using manga109s data + 1.5 million synthetic data, I could get a cer 0.1017, which is comparable to cer 0.1056 of original author's model.

|   method   |  epoches   |  validation cer    |
| ---- | ---- | ---- |
|   only manga109s data, no pretraining weights(cold start), deit tiny + bert (2 decoder layers)   | 100+     |   around 20%   |
|   only manga109s data, with pretraining weights, deit tiny + bert(2 decoder layers)  |  100+    |  around 13%    |
|   only manga109s data, with pretraining weights, deit tiny + bert(12 decoder layers) |  100+    |  around 13%    |
|   manga109s data + 1.5 million synthetic data, with pretraining weights, deit tiny + bert(2 decoder layers)  |  15+    |  around 10%    |

- Notes
  - update to get_model: it's in my_get_model.py, more straightforward.
  - see Examples section for comparing my trained model and the model trained by original author
  - only using manga109s data is not enough, the dataset is too small, i.e., only 102253 data samples, I'll need to try using synthetic data.
  - I have manually created a tiny dataset, using Jump manga 2025-3 episode, only 501 items, the cer is 0.1643 on this dataset, original author's cer is 0.1204, still need to check the reason.

# Manga OCR

Optical character recognition for Japanese text, with the main focus being Japanese manga.
It uses a custom end-to-end model built with Transformers' [Vision Encoder Decoder](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) framework. 

Manga OCR can be used as a general purpose printed Japanese OCR, but its main goal was to provide a high quality
text recognition, robust against various scenarios specific to manga:
- both vertical and horizontal text
- text with furigana
- text overlaid on images
- wide variety of fonts and font styles
- low quality images

Unlike many OCR models, Manga OCR supports recognizing multi-line text in a single forward pass,
so that text bubbles found in manga can be processed at once, without splitting them into lines.

See also:
- [Poricom](https://github.com/bluaxees/Poricom), a GUI reader, which uses manga-ocr
- [mokuro](https://github.com/kha-white/mokuro), a tool, which uses manga-ocr to generate an HTML overlay for manga
- [Xelieu's guide](https://rentry.co/lazyXel), a comprehensive guide on setting up a reading and mining workflow with manga-ocr/mokuro (and many other useful tips)
- Development code, including code for training and synthetic data generation: [link](manga_ocr_dev)
- Description of synthetic data generation pipeline + examples of generated images: [link](manga_ocr_dev/synthetic_data_generator)

# Installation

You need Python 3.6 or newer. Please note, that the newest Python release might not be supported due to a PyTorch dependency, which often breaks with new Python releases and needs some time to catch up.
Refer to [PyTorch website](https://pytorch.org/get-started/locally/) for a list of supported Python versions.

Some users have reported problems with Python installed from Microsoft Store. If you see an error:
`ImportError: DLL load failed while importing fugashi: The specified module could not be found.`,
try installing Python from the [official site](https://www.python.org/downloads).

If you want to run with GPU, install PyTorch as described [here](https://pytorch.org/get-started/locally/#start-locally),
otherwise this step can be skipped.

## Troubleshooting

- `ImportError: DLL load failed while importing fugashi: The specified module could not be found.` - might be because of Python installed from Microsoft Store, try installing Python from the [official site](https://www.python.org/downloads)
- problem with installing `mecab-python3` on ARM architecture - try [this workaround](https://github.com/kha-white/manga-ocr/issues/16)

# Usage

## Python API

```python
from manga_ocr import MangaOcr

mocr = MangaOcr()
text = mocr('/path/to/img')
```

or

```python
import PIL.Image

from manga_ocr import MangaOcr

mocr = MangaOcr()
img = PIL.Image.open('/path/to/img')
text = mocr(img)
```

## Running in the background

Manga OCR can run in the background and process new images as they appear.

You might use a tool like [ShareX](https://getsharex.com/) or [Flameshot](https://flameshot.org/) to manually capture a region of the screen and let the
OCR read it either from the system clipboard, or a specified directory. By default, Manga OCR will write recognized text to clipboard,
from which it can be read by a dictionary like [Yomichan](https://github.com/FooSoft/yomichan).

Clipboard mode on Linux requires `wl-copy` for Wayland sessions or `xclip` for X11 sessions. You can find out which one your system needs by running `echo $XDG_SESSION_TYPE` in the terminal.

Your full setup for reading manga in Japanese with a dictionary might look like this:

capture region with ShareX -> write image to clipboard -> Manga OCR -> write text to clipboard -> Yomichan

https://user-images.githubusercontent.com/22717958/150238361-052b95d1-0152-485f-a441-48a957536239.mp4

- To read images from clipboard and write recognized texts to clipboard, run in command line:
    ```commandline
    manga_ocr
    ```
- To read images from ShareX's screenshot folder, run in command line:
    ```commandline
    manga_ocr "/path/to/sharex/screenshot/folder"
    ```
Note that when running in the clipboard scanning mode, any image that you copy to clipboard will be processed by OCR and replaced
by recognized text. If you want to be able to copy and paste images as usual, you should use the folder scanning mode instead
and define a separate task in ShareX just for OCR, which saves screenshots to some folder without copying them to clipboard.

When running for the first time, downloading the model (~400 MB) might take a few minutes.
The OCR is ready to use after `OCR ready` message appears in the logs.

- To see other options, run in command line:
    ```commandline
    manga_ocr --help
    ```

If `manga_ocr` doesn't work, you might also try replacing it with `python -m manga_ocr`.

## Usage tips

- OCR supports multi-line text, but the longer the text, the more likely some errors are to occur.
  If the recognition failed for some part of a longer text, you might try to run it on a smaller portion of the image.
- The model was trained specifically to handle manga well, but should do a decent job on other types of printed text,
  such as novels or video games. It probably won't be able to handle handwritten text though. 
- The model always attempts to recognize some text on the image, even if there is none.
  Because it uses a transformer decoder (and therefore has some understanding of the Japanese language),
  it might even "dream up" some realistically looking sentences! This shouldn't be a problem for most use cases,
  but it might get improved in the next version.

# Examples

Here are some cherry-picked examples showing the capability of the model. 

| image                | Manga OCR result | My trained model result |
|----------------------|------------------| ------------------| 
| ![](assets/examples/00.jpg) | 素直にあやまるしか | 素直にあやまるしか | 
| ![](assets/examples/01.jpg) | 立川で見た〝穴〟の下の巨大な眼は： | 立川で見た”穴”の下の巨大な眼は．．．|
| ![](assets/examples/02.jpg) | 実戦剣術も一流です | 実験剣術も一緒です |
| ![](assets/examples/03.jpg) | 第３０話重苦しい闇の奥で静かに呼吸づきながら | 第３０話重告しい闇の裏で静かに呼ぶつきましょうね |
| ![](assets/examples/04.jpg) | よかったじゃないわよ！何逃げてるのよ！！早くあいつを退治してよ！ |  よかったじゃないわよ！！何逃げてるのよ！？早くあいつを退治してよ！ |
| ![](assets/examples/05.jpg) | ぎゃっ | ぎゃっ | 
| ![](assets/examples/06.jpg) | ピンポーーン | ピンポーーン| 
| ![](assets/examples/07.jpg) | ＬＩＮＫ！私達７人の力でガノンの塔の結界をやぶります | ＬＩＮＫ！私達２人の方でカンジの皆の結界をやぶります | 
| ![](assets/examples/08.jpg) | ファイアパンチ | ファイアパンチ | 
| ![](assets/examples/09.jpg) | 少し黙っている | 少し黙っている | 
| ![](assets/examples/10.jpg) | わかるかな〜？ | わかるかな～？ |
| ![](assets/examples/11.jpg) | 警察にも先生にも町中の人達に！！ | 警察にも先生にも前中の人達に！！ |

# Contact
For any inquiries, please feel free to contact me at kha-white@mail.com

# Acknowledgments

This project was done with the usage of:
- [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
- [CC-100](https://data.statmt.org/cc-100/) dataset
