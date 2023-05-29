# ConvGenVisMo

# Installation
Run the following in your shell after activating environment:
`pip install -r requirements.txt`


# Dataset
The ConvGenVisMo dataset is available in the `dataset` directory. The `dataset.jsonl` file contains all the conversations with their respective labels; the corresponding images can be found in `dataset/images`.
The images generated by Dreamstudio are available in the `generated_images_dreamstudio` directory.

# Usage
To generate ChatGPT summaries for the conversations in the dataset, run `chatgpt_summaries.ipynb`.
To generate DreamStudio images for the generated summaries, run `dreamstudio_imagegen.ipynb`.
To compute and plot metrics comparing ground truth images with DreamStudio generated images, run `compute_metrics.ipynb` and `plot_metrics.ipynb`.

We express our gratitude to the following individuals who generously contributed by sharing their captured photographs with us. We extend our sincerest thanks for their invaluable support.

[Dr. Maryam Tavakkoli](https://instagram.com/tavakkoli56?igshid=MzRlODBiNWFlZA==),
[Dr. Mahdie Niknezhad](https://instagram.com/doctor_niknezhad_atfal?igshid=MzRlODBiNWFlZA==),
[Ms. Zahra Hoseinpoor](https://instagram.com/zahra.hoseinpoor61?igshid=MzRlODBiNWFlZA==),
[Ms. Haniye Hayati](https://instagram.com/haniye_hayati?igshid=MzRlODBiNWFlZA==),
[Ms. Niloufar Khoshpasand](https://instagram.com/niloufar.fns?igshid=MzRlODBiNWFlZA==),
[Ms. Mahnaz Razavi](https://instagram.com/kardely_art?igshid=MzRlODBiNWFlZA==),
[Ms. Maryam Nosrati](https://www.linkedin.com/in/maryam-nosrati-1aa672233)


