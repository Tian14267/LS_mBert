# LS-mBERT



### Configuration Environment


- Install Conda:  https://docs.conda.io/en/latest/miniconda.html
- Found Your Conda environment:

``` sh
conda create -n LsMbert python=3.8
conda activate LsMbert
pip install -r requirements.txt
```


### Data Preparation
The next step is to download the data. To this end, first create a `download` folder with `mkdir -p download` in the root 
of this project. You then need to manually download `panx_dataset` (for NER) from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN)
(note that it will download as `AmazonPhotos.zip`) to the download directory. Finally, run the following command to 
download the remaining datasets:

```bash
bash scripts/download_data.sh
```

To get the POS-tags and dependency parse of input sentences, we use UDPipe. Go to the 
[udpipe](https://github.com/wasiahmad/Syntax-MBERT/tree/main/udpipe) directory and run the task-specific scripts -
`[xnli.sh|pawsx.sh|panx.sh|mtop.sh]`.

Notice：`Data Preparation` is same as [here](https://github.com/wasiahmad/Syntax-MBERT)


### Training 
#### Language Map
You can download files from [here](https://pan.baidu.com/s/16xabmTo9_bU0HES20o6v4w?pwd=oor6),
and put it to file directory.\
Or, you can create it yourself with your own data.

#### Text Classification
```bash
# for PAWS-X
sh run_pawsx.sh

# for XNLI
sh run_xnli.sh 
```

#### Named Entity Recognition
```bash
sh run_ner.sh
```


#### Task-oriented Semantic Parsing

```bash
sh mtop.sh
```


### Evaluation 
Take `mtop` as an example.\
**First**:  Download the relevant data from Baidu link [here](https://pan.baidu.com/s/1Jwf34v42yQaazN68VoHd6Q?pwd=1mow)\
**Second**:  Download the model from Baidu link [here](https://pan.baidu.com/s/1uiJRy_qn57F0SNIhw6-jZA?pwd=aare)
    
```
export CUDA_VISIBLE_DEVICES=0
Output_dir="./outputs/mtop_paper"
python mtop_paper.py \
    --data_dir "./download/mtop_udpipe_processed" \
    --model_name_or_path "./outputs/mtop_model" \
    --intent_labels "./download/mtop_udpipe_processed/intent_label.txt" \
    --slot_labels "./download/mtop_udpipe_processed/slot_label.txt" \
    --do_test \
    --train_langs "en" \
    --output_dir $Output_dir
```

The test of other task , or the other ideas if you have ,You can try it yourself.

