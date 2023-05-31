
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

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""@misc{xu2023tool,
      title={On the Tool Manipulation Capability of Open-source Large Language Models}, 
      author={Qiantong Xu and Fenglu Hong and Bo Li and Changran Hu and Zhengyu Chen and Jian Zhang},
      year={2023},
      eprint={2305.16504},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
}"""


block = gr.Blocks()

with block:
    gr.Markdown(
        """# Toolbench Leaderboard
    
    Welcome to the leaderboard of the ToolBench! 🏆 
    This is a community where participants create language models and action generation algorithms to generate API function calls based goals described in natural lanugage!
    Please refer to [our paper](https://arxiv.org/abs/2305.16504) for more details and join our [Discord](https://discord.com/invite/JehFG5HXKb) for further discussion.
    The [evaluation suite](https://github.com/sambanova/toolbench/) is now alive on Github.
    """
    )

    with gr.Row():
        with gr.Accordion("Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON_TEXT,
                label=CITATION_BUTTON_LABEL,
                elem_id="citation-button",
            ).style(show_copy_button=True)

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