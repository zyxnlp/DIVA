# DIVA
Repository for the paper "[Causal Inference from Text: Unveiling Interactions between Variables](https://arxiv.org/abs/2311.05286)" appear in Findings of EMNLP 2023.

### Setup:  
* `git clone https://github.com/zyxnlp/DIVA.git`
* `cd DIVA`
* `conda create -n DIVA python=3.9.7` 
* `conda activate DIVA` or `source activate DIVA`
* `conda install pytorch==1.11.0 -c pytorch`
* `pip install -r requirements.txt` 

### Datasets:  
* Downlaod datasets from [here](https://drive.google.com/file/d/1Cphjnh1VGnTeA76hWohOG1RfeQLT1BLH/view?usp=sharing)
* Unzip the .zip file under the folder `DIVA/`

### Pre-trained model:  
* Downlaod finbert model from [here](https://drive.google.com/file/d/1zLf9MqrhC2eBkK9WVH5puTrhz_SF1F9-/view?usp=sharing)
* Unzip the .zip file under the folder `DIVA/`

### Running:
```
./run.sh
```

### Citation
If you find our work or the code useful, please cite our paper:
```
@inproceedings{zhou2023causal,
 title={Causal Inference from Text: Unveiling Interactions between Variables},
 author={Zhou, Yuxiang and He, Yulan},
 booktitle={Findings of EMNLP},
 year={2023},
 url={https://arxiv.org/pdf/2311.05286.pdf}
}
```