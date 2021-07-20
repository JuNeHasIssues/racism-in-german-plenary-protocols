import torch
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import MLFlowLogger, set_all_seeds, initialize_device_settings


def main():
    # For detailed analytics!
    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name="Rassimus in Plenarprotokollen", run_name="Exp3: MoreData: LR-> 2e-5, Epo-> 2, BS-> 8")

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    n_epochs = 2
    batch_size = 8  # mehr als 16 braucht ggf. zu viel computing power
    evaluate_every = 100
    learning_rate = 2e-5

    # Here we initialize a tokenizer that will be used for preprocessing text
    # This is the BERT Tokenizer which uses the byte pair encoding method.
    # It is loaded with a German model
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="bert-base-german-cased",
        do_lower_case=False)

    data_dir = "experiments/more_data/data"
    trainfile = "train_moredata.csv"
    testfile = "test_moredata.csv"

    # Processor is customized to my own data
    LABEL_LIST = ["OTHER", "RACISM"]
    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=512,  # Maximum für BERT = 512
                                            data_dir=data_dir,
                                            label_list=LABEL_LIST,
                                            metric="f1_macro",
                                            label_column_name="LABEL",
                                            text_column_name="TEXT",
                                            delimiter=";",
                                            train_filename=trainfile,
                                            test_filename=testfile,
                                            dev_filename=None,
                                            dev_split=0.1)  # nimmt 10% des train sets zur Erstellung des dev sets

    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size
    )

    # loading the pretrained BERT base cased model
    MODEL_NAME = "bert-base-german-cased"
    language_model = LanguageModel.load(MODEL_NAME)

    # prediction head fuer das Model, ausgerichtet auf Textclassification mit zwei Labels
    prediction_head = TextClassificationHead(num_labels=len(LABEL_LIST))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.5,
        lm_output_types=["per_sequence"],
        device=device)

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=learning_rate,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    trainer.train()

    # Model wird gespeichert, um es anschließend laden und auf Texte anwenden zu können (test_model.py)
    save_dir = "experiments/more_data/models/moredata_lr_2e-5_epo_2_bs_8"
    model.save(save_dir)
    processor.save(save_dir)


if __name__ == "__main__":
    main()




