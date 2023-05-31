
__all__ = ['block', 'make_clickable_model', 'make_clickable_user', 'get_submissions']

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, repocard

def is_duplicated(space_id:str)->None:
    card = repocard.RepoCard.load(space_id, repo_type="space")
    return getattr(card.data, "duplicated_from", None) is not None



def make_clickable_model(model_name, link=None):
    if link is None:
        link = "https://huggingface.co/" + "spaces/" + model_name
    return f'<a target="_blank" href="{link}">{model_name.split("/")[-1]}</a>'

def get_space_ids():
    api = HfApi()
    spaces = api.list_spaces(filter="making-demos")
    print(spaces)
    space_ids = [x for x in spaces]
    return space_ids


def make_clickable_user(user_id):
    link = "https://huggingface.co/" + user_id
    return f'<a  target="_blank" href="{link}">{user_id}</a>'

def get_submissions():
    submissions = get_space_ids()
    leaderboard_models = []

    for submission in submissions:
        # user, model, likes
        if not is_duplicated(submission.id):
            user_id = submission.id.split("/")[0]
            leaderboard_models.append(
                (
                    make_clickable_user(user_id),
                    make_clickable_model(submission.id),
                    submission.likes,
                )
            )

    df = pd.DataFrame(data=leaderboard_models, columns=["User", "Space", "Likes"])
    df.sort_values(by=["Likes"], ascending=False, inplace=True)
    df.insert(0, "Rank", list(range(1, len(df) + 1)))
    return df

block = gr.Blocks()

with block:
    gr.Markdown(
        """# Making Demos Leaderboard
    
    Welcome to the leaderboard for the ever-running Making Demos Event! 🏆 
    This is a community event where participants create demos of recently released machine learning models, or combine multiple demos to make cool apps!
    To attend the event, simply join us in [Discord](https://huggingface.co/join/discord), take the role #collaborate and write under one of the paper posts under #making-demos forum.
    You can add `making-demos` to tags in your Spaces' README.md's metadata section to add your demo to this leaderboard. 
    At the end of every week, winners of this leaderboard will earn special prizes! 🎁 
    """
    )

    with gr.Row():
        data = gr.components.Dataframe(
            type="pandas", datatype=["number", "markdown", "markdown", "number"]
        )
    with gr.Row():
        data_run = gr.Button("Refresh")
        data_run.click(
            get_submissions, outputs=data
        )

    block.load(get_submissions, outputs=data)

block.launch()