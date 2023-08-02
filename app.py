
__all__ = ['block', 'make_clickable_model', 'make_clickable_user', 'get_submissions']

import gradio as gr
import pandas as pd

COLUMN_NAMES = ["Model", "Tuned on ToolBench", "Avg.", "Open Weather", "The Cat API", "Home Search", "Trip Booking", "Google Sheets", "VirtualHome", "WebShop Long", "WebShop Short", "Tabletop"]
UNTUNED_MODEL_RESULTS = '''[gpt4](https://platform.openai.com/docs/models/gpt-4)                    & 93.0 & 96.0 & 97.0 & 96.7 & 62.9 & 23.0 / 23.5 & 0.0 & 0.0 & 81.0 \\
[text-davinci-003](https://platform.openai.com/docs/models/gpt-3)      & 99.0 & 98.0 & 97.0 & 89.2 & 62.9 & 31.0 / 25.1 & 0.0 & 0.0 & 66.7 \\
[gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)           & 90.0 & 92.0 & 80.0 & 85.8 & 51.4 & 20.0 / 18.9 & 0.0        & 1.8        & 33.3 \\
[text-curie-001](https://platform.openai.com/docs/models/gpt-3)          & 8.0  & 58.0 & 6.0  & 6.7  & 1.4  & 12.0 / 4.1  & 0.0        & 0.0        & 1.0  \\
[Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b) & 90.0 & 84.39 & 83.0 & 71.67 & 58.57 & 35.0 / 24.74 & 1.53 & 30.45 & 45.4 \\
[Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b) & 85.0 & 77.0 & 68.0 & 53.33 & 30.0 & 33.0 / 21.67 & 0.6 & 31.67 & 23.81 \\
[Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) & 76.0 & 83.0 & 58.0 & 33.33 & 22.86 & 25.0 / 21.49 & 0.0 & 6.92 & 14.39
[llama-65b](https://huggingface.co/huggyllama/llama-65b)     & 90.0 & 80.0 & 84.0 & 65.8 & 32.9 & 32.0 / 20.3 & 0.0 & 41.2 & 30.5 \\
[llama-30b](https://huggingface.co/huggyllama/llama-30b)              & 78.0 & 84.0 & 66.0 & 45.0 & 37.1 & 27.0 / 21.7 & 0.0 & 30.6 & 34.3 \\
[llama-13b](https://huggingface.co/huggyllama/llama-13b)                & 70.0 & 74.0 & 45.0 & 35.8 & 5.7  & 28.0 / 18.9 & 0.0 & 27.6 & 17.1 \\
[llama-13b-alpaca](https://huggingface.co/chavinlo/gpt4-x-alpaca)    & 62.0 & 43.0 & 44.0 & 40.8 & 11.4 & 1.0 / 1.6   & 0.0 & 2.7  & 9.5  \\
[starcoder](https://huggingface.co/bigcode/starcoder)                & 91.0 & 84.0 & 82.0 & 51.7 & 48.0 & 23.0 / 19.4 & 2.6 & 0.0  & 21.9 \\
[starcoderbase](https://huggingface.co/bigcode/starcoderbase)           & 90.0 & 86.0 & 79.0 & 63.3 & 42.9 & 24.0 / 16.3 & 5.8 & 23.1 & 17.1 \\
[codegen-16B-nl](https://huggingface.co/Salesforce/codegen-16B-nl)           & 51.0 & 75.0 & 37.0 & 21.7 & 7.1  & 43.0 / 18.0 & 0.0 & 0.0  & 16.2 \\
[codegen-16B-multi](https://huggingface.co/Salesforce/codegen-16B-multi)        & 56.0 & 75.0 & 47.0 & 7.5  & 21.4 & 31.0 / 14.1 & 0.0 & 0.5  & 8.6  \\
[codegen-16B-mono](https://huggingface.co/Salesforce/codegen-16B-mono)        & 63.7 & 72.0 & 52.0 & 28.3 & 31.5 & 28.0 / 15.7 & 1.5 & 6.6  & 15.2 \\
[bloomz](https://huggingface.co/bigscience/bloomz)            & 58.0 & 85.0 & 36.0 & 22.5 & 14.3 & 9.0 / 4.9   & 0.0 & 1.0  & 1.0  \\
[opt-iml-30b](https://huggingface.co/facebook/opt-iml-30b)              & 44.0 & 48.0 & 5.0  & 3.3  & 2.9  & 13.0 / 8.3  & 0.0 & 0.0  & 1.0  \\
[opt-30b](https://huggingface.co/facebook/opt-30b)                  & 46.0 & 35.0 & 2.0  & 3.3  & 8.6  & 24.0 / 11.7 & 0.0 & 0.0  & 1.0  \\
[opt-iml-1.3b](https://huggingface.co/facebook/opt-iml-1.3b)             & 20.0 & 28.0 & 0.0  & 0.0  & 4.3  & 13.0 / 3.1  & 0.0 & 0.0  & 1.0  \\
[opt-1.3b](https://huggingface.co/facebook/opt-1.3b)                 & 18.0 & 30.0 & 0.0  & 0.0  & 1.4  & 31.0 / 9.7  & 0.0 & 0.0  & 1.0  \\
[neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)                & 55.0 & 69.0 & 27.0 & 10.8 & 18.6 & 28.0 / 15.3 & 0.0 & 8.8  & 6.7  \\
[GPT-NeoXT-Chat-Base-20B](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B)  & 43.0 & 73.0 & 28.0 & 10.8 & 4.3  & 26.0 / 13.1 & 0.0 & 0.7  & 7.6  \\
[pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)               & 53.0 & 65.0 & 12.0 & 0.8  & 11.4 & 17.0 / 12.1 & 0.0 & 0.0  & 1.9  \\
[dolly-v2-12b]()            & 0.0  & 1.0  & 10.0 & 5.0  & 7.1  & 11.0 / 8.9  & 0.0 & 0.0  & 7.6  \\
[pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b)   & 41.0 & 72.0 & 8.0  & 7.5  & 4.3  & 29.0 / 14.0 & 0.0 & 0.0  & 8.6  \\
[pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)   & 49.0 & 54.0 & 7.0  & 3.3  & 12.9 & 24.0 / 14.8 & 0.0 & 0.0  & 7.6  \\
[pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)   & 37.0 & 48.0 & 4.0  & 5.0  & 10.0 & 22.0 / 10.7 & 0.0 & 5.2  & 7.6  \\
[stablelm-base-alpha-7b](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)   & 22.0 & 47.0 & 0.0  & 0.0  & 4.3  & 28.0 / 10.3 & 0.0 & 0.0  & 2.9  \\
[stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)  & 23.0 & 38.0 & 0.0  & 0.0  & 1.4  & 26.0 / 7.3  & 0.0 & 0.0  & 3.8  \\
[stablelm-base-alpha-3b](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)   & 6.0  & 28.0 & 0.0  & 0.0  & 1.4  & 29.0 / 5.3  & 0.0 & 0.0  & 1.0  \\
[stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)  & 14.0 & 31.0 & 0.0  & 0.8  & 0.0  & 8.0 / 5.6   & 0.0 & 0.0  & 1.0  \\'''
TUNED_MODEL_RESULTS='''[llama-30b-toolbench](https://huggingface.co/sambanovasystems/LLaMA-30b-toolbench)          & 100.0          & 94.0           & 87.0           & 85.8           & 2.9            & 16.0/ 24.3& 0.0            & 0.0            & 7.5        \\
[starcoder-toolbench](https://huggingface.co/sambanovasystems/starcoder-toolbench)          & 99.0       & 97.0           & 83.0           & 80.8           & 21.2        & 31.0/ 18.4& 0.0            & 0.0            & 13.9        \\
[codegen-16B-mono-toolbench](https://huggingface.co/sambanovasystems/codegen-16B-mono-toolbench)   & 97.7          & 99.0           & 82.0           & 77.5           & 19.8     & 29.0/ 17.2& 0.0            & 3.5            & 16.2                   \\'''


def parse_line(line):
    model_results = line.replace(" ", "").strip("\\").split("&")
    for i in range(1, len(model_results)):
        if i == 6:
            res = model_results[6].split('/')[-1].strip()
        else:
            res = model_results[i]
        model_results[i] = float(res)
    return model_results
    
def get_baseline_df():
    df_data = []
    
    lines = UNTUNED_MODEL_RESULTS.split("\n")
    for line in lines:
        model_results = parse_line(line)
        assert len(model_results) == 10
        avg = sum(model_results[1:]) / 9
        model_results.insert(1, avg)
        model_results.insert(1, "False")
        df_data.append(model_results)
    lines = TUNED_MODEL_RESULTS.split("\n")
    for line in lines:
        model_results = parse_line(line)
        assert len(model_results) == 10
        avg = sum(model_results[1:]) / 9
        model_results.insert(1, avg)
        model_results.insert(1, "True")
        df_data.append(model_results)
        
    print(len(df_data))
    df = pd.DataFrame(df_data, columns=COLUMN_NAMES).round(1)
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

    
    gr.Markdown(
        """In the table below, we summarize the 3-shot performance of all the models.
        We use success rate as the primary evaluation metric for most tasks, except that we report rewards on WebShop, and the Longest Common Subsequence (LCS) on VirtualHome, following the original metrics proposed by the respective authors. 
    """
    )
    with gr.Row():
        data = gr.components.Dataframe(
            type="pandas", datatype=["markdown", "markdown", "number", "number", "number", "number", "number", "number", "number", "number", "number", "number"]
        )
    with gr.Row():
        data_run = gr.Button("Refresh")
        data_run.click(
            get_baseline_df, outputs=data
        )

    block.load(get_baseline_df, outputs=data)

block.launch()