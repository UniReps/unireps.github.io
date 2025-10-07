---
layout: distill
title: Submission Guidelines
description: How to submit a blogpost to the UniReps Workshop
tags: guide, tutorial
giscus_comments: false
date: 2024-10-09
featured: true

authors:
  - name: Valentino Maiorca
    url: "https://flegyas.github.io/"
    affiliations:
      name: Sapienza University of Rome

toc:
  - name: Getting Started
    #     # if a section has subsections, you can add them as follows:
    subsections:
      - name: Fork the Repository
      - name: Clone Your Fork Locally
      - name: Preview the Website Locally
  - name: Writing Your Post
    subsections:
      - name: Create a New Post
      - name: Add Media (Optional)
      - name: Submitting Your Blogpost
  - name: Review and Publication
    subsections:
      - name: Automatic Checks
      - name: Review Process
      - name: Post-Acceptance
  - name: Questions?

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

Welcome to the **UniReps Blogpost Submission** guide! This page outlines the steps to submit your blogpost to the UniReps Workshop, where you can share your insights, spark discussions, and connect with the community.
We're excited you're interested in contributing!

To manage the blogpost track, we rely on [al-folio](https://github.com/alshedivat/al-folio), a [Jekyll](https://jekyllrb.com) theme for academic websites.
If this is all new to you, don't worry. It's simpler than it sounds and **we are available for any questions you might have**.

## Getting Started

### Fork the Repository

Start by **forking the repository**. You can do that [here](https://github.com/UniReps/UniReps). Forking creates your own copy of the project where you'll be able to freely work on your post before submitting it via a pull request (PR).

### Clone Your Fork Locally

Next, clone your new repository to your local machine so you can start editing:

```bash
git clone git@github.com:UniReps/UniReps.git
cd UniReps
```

## Create a new branch

Create a new branch specific for each post submission. We recommend this format for the branch name:

```bash
git checkout -b post/<your-post-title>
```

You could even use the GitHub web interface to directly write the blogpost, but we recommend cloning the repository for a smoother experience.

### Preview the Website Locally

We've got a few options depending on your setup. Choose what works best for you:

- **Option 1: Jekyll**  
   If you have [Jekyll](https://jekyllrb.com/) installed, run:

  ```bash
  bundle install
  bundle exec jekyll serve
  ```

- **Option 2: Docker**  
   If Docker is your thing, simply run:

  ```bash
  docker-compose up
  ```

- **Option 3: VS Code**  
   If you are a **VS Code** user, we suggest using the [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) feature, and it will take care of the environment setup for you.

All these options will start a local server, and you'll be able to preview the website at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

If you are unsure about which option to choose, we recommend starting with **Option 3** as it's the easiest to set up (assuming you are already using VS Code).
However, if you run into any issues, don't hesitate to reach out to us for help!

---

## Writing Your Post

Now to the exciting part—**writing your blogpost**!

### Topics and Styles 📝

We welcome blogposts of different nature:
- New or early-stage research results 🧑‍🔬
- Tutorial-style summaries of key methods and literature 🔍
- Opinion pieces on relevant topics in UniReps 🧠
- And more!

Remember, the goal is to share your insights, spark discussions, and connect with the community. Keep it engaging and accessible to the [broad workshop audience](https://unireps.org/2025/)!

### Create a New Post

Head over to the `_posts/` directory and create a new file following this format:

```text
_posts/YYYY-MM-DD-title.md
```

A great starting point is the template `/_posts/2024-10-09-template.md`, the original one from [Distill](https://distill.pub). Create a **copy** of it (it's important, don't directly edit it), rename it, and edit it for your post. It's already structured for you!

### Add Media (Optional)

If your post includes media (we recommend to include them to exploit the blogpost format at its best), here's how to add them:

- **Images**: Place images in the folder `assets/img/YYYY-MM-DD-title/`.
- **Plotly Interactive Figures**: Drop them into `assets/plotly/YYYY-MM-DD-title/`.
- **Citations**: Save them as a BibTeX file (`.bib`) in `assets/bibliography/YYYY-MM-DD-title.bib`.
- **Other file types**: follow the structure of the `assets/` folder, placing them in the appropriate subfolder.

To display them in your post, use the following syntax:
```html
![Your Media]({{ '/assets/complete/media/path/including.extension' | relative_url }})
```

For example, if you have an image `unireps_banner.jpeg` in the folder `assets/img/2024-10-09-guidelines/`, you would include it in your post like this:
```html
![My Image]({{ '/assets/img/2024-10-09-guidelines/unireps_banner.jpeg' | relative_url }})
```
Resulting in:
![My Image]({{ '/assets/img/2024-10-09-guidelines/unireps_banner.jpeg' | relative_url }})

For more examples on how to include media, check the [Post Template](https://unireps.org/blog/2024-10-09-template) or the [Distill Guide](https://distill.pub/guide/).

---


## Submitting Your Blogpost

Ready to submit? Follow these steps:

### Push Your Changes

Push the changes from your local machine to your forked repository:

As an example:

```bash
git add .
git commit -m "Blogpost submission: <title>"
git push origin master
```

### Open a Pull Request

Now, head to your forked repository on GitHub and open a **New Pull Request**. Ensure the title reflects the topic of your blogpost, and double-check the description is clear and concise.

---

## Review and Publication

Here's what happens next:

### Automatic Checks

The repository is set up to run automatic checks on your submission so be sure to:

1. Only create/edit new files, never edit existing ones from the original repository.
2. Create one PR per blogpost submission (i.e., only one new .md file under the `_posts` directory).
3. Place all the files in the appropriate directories with the submission pattern mentioned above.

If everything looks good, the PR will be marked as "Ready for Review" and you will also get a preview link to see how your blogpost will look like on the website.

### Review Process

Your blogpost will be reviewed based on the live content (ignoring the commit history or previous drafts). The review process will focus on the following aspects:

- **Content**: Is the content relevant for the community, insightful, and engaging?
- **Structure**: Is the blogpost well-structured and easy to follow?
- **Media**: Are the images, plots, and other media elements well-integrated?
- **Style**: Is the writing clear, concise, and correct?

### Post-Acceptance

Once your blogpost is accepted, it will be merged into the main repository and published on the website. You will be notified via email and your blogpost will be shared on the workshop's social media channels.

---

## Questions?

If you need any help or run into any issues, don't hesitate to reach out to us:

- Open an issue on the [GitHub repository](https://github.com/UniReps/UniReps)
- Email: unireps-workshop [at] university [dot] org

🔵 We're here to help! 🔴
