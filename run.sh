echo "#############Causal Estimation: Political Risk -> Stock Volatility#############"
python main.py --interest prisk --task vol --gpu 0 --num_train_epochs 30 --batch_size 86 --seed 19

echo "#############Causal Estimation: Sentiment -> Stock Volatility#############"
python main.py --interest senti --task vol --gpu 0 --num_train_epochs 30 --batch_size 86 --seed 19


echo "#############Causal Estimation: Political Risk -> Stock Movement#############"
python main.py --interest prisk --task mov --gpu 0 --num_train_epochs 30 --batch_size 86 --seed 19

echo "#############Causal Estimation: Sentiment -> Stock Movement#############"
python main.py --interest senti --task mov --gpu 0 --num_train_epochs 30 --batch_size 86 --seed 19
