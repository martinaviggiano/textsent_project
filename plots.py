import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_freq_labels(data, template="plotly"):
    X = ["Non Hate Speech", "Hate Speech"]
    Y = data["label"].value_counts().values

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=X,
            y=Y,
            text=Y,
            textposition="auto",
            marker_color=["lightblue", "royalblue"],
            hovertemplate="Label: %{x} <br>Count: %{y}",
        )
    )

    fig.update_layout(
        title="Labels frequency",
        xaxis_title="Labels",
        yaxis_title="Counts",
        template=template,
    )

    return fig


def plot_word_hist(data, template="plotly"):
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data.word_count_before.values,
            marker_color="royalblue",
            name="Before cleaning",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=data.word_count.values,
            marker_color="lightblue",
            name="After cleaning",
        )
    )

    fig.update_layout(
        title="Words distribution",
        xaxis_title="Number of words",
        yaxis_title="Number of sentences",
        barmode="stack",
        template=template,
    )

    fig.update_xaxes(range=[0, 50])

    return fig


def plot_most_common_words(df, template="plotly"):
    X = df.words
    Y = df.freq

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=X,
            y=Y,
            hovertemplate="Word: %{x} <br>Count: %{y}",
            marker_color= "royalblue",
        )
    )

    fig.update_layout(
        title="Top 20 most common Words in the entire dataset ",
        xaxis_title="Word",
        yaxis_title="Count",
        xaxis_tickangle=290,
        template=template,
    )

    return fig


def plot_top_20_pos(df, x_col="", title="", template="plotly"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df.Freq_Tot,
            name="Freq. tot.",
            yaxis="y",
            offsetgroup=1,
            marker_color="lightblue",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df.Freq_Hate_Speech,
            name="Freq. Hate Speech",
            yaxis="y2",
            offsetgroup=2,
            marker_color="royalblue",
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text=x_col, tickangle = 290)
    fig.update_yaxes(title_text="Count", secondary_y=False)

    fig.update_layout(
        title=title, template=template, yaxis2=dict(overlaying="y", side="right")
    )

    fig.update_layout(barmode="group")

    return fig
