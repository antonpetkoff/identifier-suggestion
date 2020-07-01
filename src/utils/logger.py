import os
import json
import wandb
import numpy as np

from src.visualization.plot import plot_attention_weights


class Logger:
    def __init__(
        self,
        experiment_config,
        wandb_save_dir = None,
        image_save_dir = None,
        data_save_dir = None,
    ):
        self.wandb_enabled = False
        self.image_save_dir = image_save_dir
        self.data_save_dir = data_save_dir

        self.log_message('Initializing logger')

        if wandb_save_dir:
            self.wandb_enabled = True

            wandb.init(dir = wandb_save_dir, config = experiment_config)

        if self.image_save_dir:
            os.makedirs(self.image_save_dir, exist_ok = True)


    def log_message(self, message, *params):
        print(message, *params)


    def log_data(self, data):
        if self.wandb_enabled:
            wandb.log(data)
        else:
            print(data)


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


    def log_plot(self, plt, save_name = None):
        if self.wandb_enabled:
            wandb.log({ f'attention_heatmap_{id}': plt })

        if self.image_save_dir and save_name:
            plt.savefig(fname = os.path.join(self.image_save_dir, save_name))


    def log_attention_heatmap(
        self,
        attention_weights,
        input_tokens,
        output_tokens,
        save_name = None
    ):
        matrix_values = attention_weights[:len(input_tokens), :len(output_tokens)]

        if self.wandb_enabled:
            wandb.log({
                save_name: wandb.plots.HeatMap(
                    x_labels = output_tokens,
                    y_labels = input_tokens,
                    matrix_values = matrix_values,
                    show_text = False
                )
            })

        if self.image_save_dir and save_name:
            plt = plot_attention_weights(
                matrix_values,
                input_tokens,
                output_tokens,
            )

            plt.savefig(
                fname = os.path.join(self.image_save_dir, save_name + '.png'),
                bbox_inches='tight'
            )

            plt.close()

        if self.data_save_dir:
            # save the data itself as JSON for further plotting afterwards
            with open(os.path.join(self.data_save_dir, save_name + '.json'), mode='w') as f:
                f.write(json.dumps({
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'weights': matrix_values.tolist(),
                }))

    def persist_data(self, data_path):
        if self.wandb_enabled:
            wandb.save(data_path)
