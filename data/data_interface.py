import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoConfig,DataCollatorForLanguageModeling,DataCollatorWithPadding
from datasets import load_dataset


class DataMovement(pl.LightningDataModule):
    def __init__(self, args:Namespace):
        super().__init__()
        self.args = args
        self.num_workers = args.num_workers
        self.encoder_model = f"{self.args.bert_folder}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self.max_length = 512
        self.batch_size = args.batch_size
        self.use_mlm = bool(args.use_mlm)

        if self.use_mlm:
            self.mlm_probability = args.mlm_probability
            self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability)
        else:
            self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.load_datasets()

    def tokenize_function_mlm(self,examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        examples["treatment_id"] = [line for line in examples["T_prisk"]] if self.args.interest == "prisk" else [line for line in examples["T_senti"]]

        examples["outcome_sim"] = [line for line in
                                   examples["Y_bvol_prisk_sim"]] if self.args.interest == "prisk" else [line for
                                                                                                        line in
                                                                                                        examples[
                                                                                                            "Y_bvol_senti_sim"]]
        examples["outcome_ctf"] = [line for line in
                                       examples["Y_bvol_prisk_sim_ctf"]] if self.args.interest == "prisk" else [line
                                                                                                                for
                                                                                                                line
                                                                                                                in
                                                                                                                examples[
                                                                                                                    "Y_bvol_senti_sim_ctf"]]

        return self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )


    def tokenize_function(self,examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        examples["treatment_id"] = [line for line in examples["T_prisk"]] if self.args.interest == "prisk" else [line for line in examples["T_senti"]]

        examples["outcome_sim"] = [line for line in
                                   examples["Y_bvol_prisk_sim"]] if self.args.interest == "prisk" else [line for
                                                                                                        line in
                                                                                                        examples[
                                                                                                            "Y_bvol_senti_sim"]]
        examples["outcome_ctf"] = [line for line in
                                       examples["Y_bvol_prisk_sim_ctf"]] if self.args.interest == "prisk" else [line
                                                                                                                for
                                                                                                                line
                                                                                                                in
                                                                                                                examples[
                                                                                                                    "Y_bvol_senti_sim_ctf"]]

        return self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=self.max_length ,
            return_special_tokens_mask=True,
        )


    def load_datasets(self) -> None:
        """
        Prepare data
        """
        data_files = {"train": "train.json", "val": "val.json", "test": "test.json"}
        raw_datasets = load_dataset(self.args.data_dir, data_files=data_files)
        remove_list=["text"]
        if self.args.interest=="prisk":
            remove_list = ["text","T_senti","Y_vol_prisk_sim","Y_vol_prisk_sim_ctf","Y_vol_senti_sim","Y_vol_senti_sim_ctf","Y_bvol_senti_sim","Y_bvol_senti_sim_ctf","Y_vol3_real","Y_vol7_real","Y_vol15_real","Y_vol30_real"]
        elif self.args.interest=="senti":
            remove_list = ["text","T_prisk","Y_vol_prisk_sim","Y_vol_prisk_sim_ctf","Y_vol_senti_sim","Y_vol_senti_sim_ctf","Y_bvol_prisk_sim","Y_bvol_prisk_sim_ctf","Y_vol3_real","Y_vol7_real","Y_vol15_real","Y_vol30_real"]

        self.processed_datasets = raw_datasets.map(self.tokenize_function_mlm if self.use_mlm else self.tokenize_function,
                                                           batched=True, num_proc=2, load_from_cache_file=True,
                                                           remove_columns=remove_list)

    def setup(self, stage: str = None):
        remove_col = []
        if self.args.interest == "prisk":
            remove_col = ["T_prisk", "Y_bvol3_real", "Y_bvol7_real", "Y_bvol15_real", "Y_bvol30_real",
                          "Y_bvol_prisk_sim", "Y_bvol_prisk_sim_ctf"]
        elif self.args.interest == "senti":
            remove_col = ["T_senti", "Y_bvol3_real", "Y_bvol7_real", "Y_bvol15_real", "Y_bvol30_real",
                          "Y_bvol_senti_sim", "Y_bvol_senti_sim_ctf"]

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.trainset = self.processed_datasets["train"].remove_columns(remove_col)
            self.valset = self.processed_datasets["val"].remove_columns(remove_col)

        if stage == "val":
            self.valset = self.processed_datasets["val"].remove_columns(remove_col)


        if stage == "test" :
            self.testset = self.processed_datasets["test"].remove_columns(remove_col)



    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.data_collator)



    def val_dataloader(self)-> DataLoader:
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.data_collator)

    def test_dataloader(self)-> DataLoader:
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.data_collator)



class DataVolatility(pl.LightningDataModule):
    def __init__(self, args:Namespace):
        super().__init__()
        self.args = args
        self.num_workers = args.num_workers
        self.encoder_model = f"{self.args.bert_folder}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self.max_length = 512
        self.batch_size = args.batch_size
        self.use_mlm = bool(args.use_mlm)

        if self.use_mlm:
            self.mlm_probability = args.mlm_probability
            self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability)
        else:
            self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.load_datasets()

    def tokenize_function_mlm(self,examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        examples["treatment_id"] = [line for line in examples["T_prisk"]] if self.args.interest == "prisk" else [line for line in examples["T_senti"]]


        examples["outcome_sim"] = [line for line in
                                   examples["Y_vol_prisk_sim"]] if self.args.interest == "prisk" else [line for
                                                                                                        line in
                                                                                                        examples[
                                                                                                            "Y_vol_senti_sim"]]
        examples["outcome_ctf"] = [line for line in
                                       examples["Y_vol_prisk_sim_ctf"]] if self.args.interest == "prisk" else [line
                                                                                                                for
                                                                                                                line
                                                                                                                in
                                                                                                                examples[
                                                                                                                    "Y_vol_senti_sim_ctf"]]

        return self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )


    def tokenize_function(self,examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        examples["treatment_id"] = [line for line in examples["T_prisk"]] if self.args.interest == "prisk" else [line for line in examples["T_senti"]]


        examples["outcome_sim"] = [line for line in
                                   examples["Y_vol_prisk_sim"]] if self.args.interest == "prisk" else [line for
                                                                                                        line in
                                                                                                        examples[
                                                                                                            "Y_vol_senti_sim"]]
        examples["outcome_ctf"] = [line for line in
                                       examples["Y_vol_prisk_sim_ctf"]] if self.args.interest == "prisk" else [line
                                                                                                                for
                                                                                                                line
                                                                                                                in
                                                                                                                examples[
                                                                                                                    "Y_vol_senti_sim_ctf"]]

        return self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )


    def load_datasets(self) -> None:
        """
        Prepare data
        """
        data_files = {"train": "train.json", "val": "val.json", "test": "test.json"}
        raw_datasets = load_dataset(self.args.data_dir, data_files=data_files)
        remove_list=["text"]

        if self.args.interest=="prisk":
            remove_list = ["text","T_senti","Y_bvol_prisk_sim","Y_bvol_prisk_sim_ctf","Y_bvol_senti_sim","Y_bvol_senti_sim_ctf","Y_vol_senti_sim","Y_vol_senti_sim_ctf","Y_bvol3_real","Y_bvol7_real","Y_bvol15_real","Y_bvol30_real"]
        elif self.args.interest=="senti":
            remove_list = ["text","T_prisk","Y_bvol_prisk_sim","Y_bvol_prisk_sim_ctf","Y_bvol_senti_sim","Y_bvol_senti_sim_ctf","Y_vol_prisk_sim","Y_vol_prisk_sim_ctf","Y_bvol3_real","Y_bvol7_real","Y_bvol15_real","Y_bvol30_real"]

        self.processed_datasets = raw_datasets.map(self.tokenize_function_mlm if self.use_mlm else self.tokenize_function,
                                                           batched=True, num_proc=2, load_from_cache_file=True,
                                                           remove_columns=remove_list)

    def setup(self, stage: str = None):
        remove_col = []

        if self.args.interest == "prisk":
            remove_col = ["T_prisk", "Y_vol3_real", "Y_vol7_real", "Y_vol15_real", "Y_vol30_real",
                          "Y_vol_prisk_sim", "Y_vol_prisk_sim_ctf"]
        elif self.args.interest == "senti":
            remove_col = ["T_senti", "Y_vol3_real", "Y_vol7_real", "Y_vol15_real", "Y_vol30_real",
                          "Y_vol_senti_sim", "Y_vol_senti_sim_ctf"]
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.trainset = self.processed_datasets["train"].remove_columns(remove_col)
            self.valset = self.processed_datasets["val"].remove_columns(remove_col)

        if stage == "test" :

            self.testset = self.processed_datasets["test"].remove_columns(remove_col)



    def train_dataloader(self) -> DataLoader:

        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.data_collator)


    def val_dataloader(self)-> DataLoader:
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.data_collator)

    def test_dataloader(self)-> DataLoader:
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.data_collator)










if __name__=='__main__':
    parser = ArgumentParser()
    from main import parse_arguments
    parser = parse_arguments(parser)
    args = parser.parse_args()

    data_moudle = DataMovement(args)




