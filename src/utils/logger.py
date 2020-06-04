import os
import wandb
import numpy as np

from src.visualization.plot import plot_attention_weights


class Logger:
    def __init__(
        self,
        experiment_config,
        wandb_save_dir = None,
        image_save_dir = None
    ):
        self.wandb_enabled = False
        self.image_save_dir = image_save_dir

        self.log_message('Initializing logger')

        if wandb_save_dir:
            self.wandb_enabled = True

            wandb.init(dir = wandb_save_dir, config = experiment_config)

        if self.image_save_dir:
            os.makedirs(self.image_save_dir, exist_ok = True)


    def log_message(self, message, *params):
        print(message, *params)


    def log_data(self, data):
        print(data)

        if self.wandb_enabled:
            wandb.log(data)


    def log_examples_table(
        self,
        input_texts,
        predicted_texts,
        expected_texts
    ):
        # TODO: save table to disk?

        if self.wandb_enabled:
            data = np.stack([
                input_texts,
                predicted_texts,
                expected_texts
            ], axis=1).tolist()

            examples_table = wandb.Table(
                data = data,
                columns = ['Input', 'Predicted', 'Actual']
            )

            wandb.log({ 'examples': examples_table })
        else:
            print(predicted_texts)


    def log_attention_heatmap(
        self,
        attention_weights,
        input_tokens,
        output_tokens,
        save_name = None
    ):
        if self.wandb_enabled:
            wandb.log({
                'attention_heatmap': wandb.plots.HeatMap(
                    x_labels = output_tokens,
                    y_labels = input_tokens,
                    matrix_values = attention_weights,
                    show_text = False
                )
            })

        if self.image_save_dir and save_name:
            plot_attention_weights(
                attention_weights,
                input_tokens,
                output_tokens,
            ).savefig(
                fname = os.path.join(self.image_save_dir, save_name)
            )


    def persist_data(self, data_path):
        if self.wandb_enabled:
            wandb.save(data_path)
