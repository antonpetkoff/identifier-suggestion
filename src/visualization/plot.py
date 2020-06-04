import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_attention_weights(
    attention_weights,
    input_tokens,
    output_tokens,
):
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention_weights, cmap = 'viridis')

    fontdict = {
        'fontsize': 14,
    }

    # matplotlib offsets the tokens by one position => the empty string
    ax.set_xticklabels([''] + output_tokens, fontdict = fontdict, rotation = 90)
    ax.set_yticklabels([''] + input_tokens, fontdict = fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator())

    return plt