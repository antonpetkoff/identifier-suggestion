import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_attention_weights(
    attention_weights,
    input_tokens,
    output_tokens,
):
    # TODO: scale the aspect ratio based on the sequence lengths?
    #  how will this affect the view experience of an example per epoch
    fig = plt.figure(figsize = (10, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention_weights, cmap = 'viridis')

    # TODO: scale the font based on the sequence lengths?
    fontdict = {
        'fontsize': 8 if len(input_tokens) > 100 else 10,
    }

    # matplotlib offsets the tokens by one position => the empty string
    ax.set_xticklabels([''] + output_tokens, fontdict = fontdict, rotation = 90)
    ax.set_yticklabels([''] + input_tokens, fontdict = fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator())

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    return plt