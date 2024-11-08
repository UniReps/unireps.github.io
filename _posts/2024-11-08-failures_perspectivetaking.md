---
layout: distill
title: Failures in Perspective-Taking of Multimodal AI Systems
description: An investigation into the spatial reasoning abilities of multimodal LLMs.
tags: distill formatting
giscus_comments: true
date: 2024-11-08
featured: true

authors:
  - name: Bridget Leonard
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: Psychology, University of Washington
  - name: Kristin Woodard
    # url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: Psychology, University of Washington
  - name: Scott O. Murray
    # url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: Psychology, University of Washington

bibliography: 2024-11-08-failures_perspectivetaking.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Interactive Plots
  - name: Layouts
  - name: Other Typography?

---

## Introduction

Visual perspective-taking, or the ability to mentally simulate a viewpoint other than one's own, is a critical aspect of spatial cognition. It allows us to understand the relationship between objects and how we might have to manipulate a scene to align with our perspective, which is essential for tasks like navigation and social interaction. Although past research has examined AI spatial cognition, it lacks the specificity found in human spatial cognition studies where processes are broken down into sub-components for more precise measurement and interpretation. In cognitive psychology, established tasks are carefully controlled to isolate specific variables, reducing bias and alternative strategies for task performance. By applying these established methods, we can evaluate AI spatial cognition more rigorously, beginning with perspective-taking. The rich human literature on these spatial skills provides a valuable benchmark, allowing us to compare model performance against the human developmental timeline and identify key areas for future research and model improvement.

---

## Background on Perspective-Taking

Perspective-taking is a cornerstone of human spatial reasoning. For multimodal models to function as effective cognitive systems and daily assistants, they must develop robust perspective-taking abilities. In the human developmental literature, perspective-taking has been stratified into two levels. Level 1 refers to knowing that a person may be able to see something another person does not, and it appears fully developed by the age of two <d-cite key="moll2006level1"></d-cite>. A common Level 1 task might ask if an object is viewable (or positioned to the front or back) of a person or avatar in a scene. Level 2 refers to the ability to represent how a scene would look from a different perspective, often measured by having subjects assess the spatial relationship between objects. Although success on some simple Level 2 tasks is first seen around age 4 <d-cite key="newcombe1992children"></d-cite>, Level 2 perspective-taking continues to develop into middle childhood <d-cite key="surtees2012egocentrism"></d-cite> and even into young adulthood <d-cite key="dumontheil2010online"></d-cite>.

A more specific cognitive process, mental rotation, where one imagines an object or scene rotating in space to align with a perspective, plays an important role in perspective-taking. Surtees et al. <d-cite key="surtees2013similarities"></d-cite> experimentally manipulated Level 1 and Level 2 perspective-taking by presenting participants with tasks where they viewed numbers or blocks relative to an avatar. Different stimuli were used to elicit visual and spatial judgments, like whether the number was a "6" or a "9" from the person's perspective, or if the block was to the person's right or left. Level 1 tasks involved indicating whether the number/block was visible to the avatar, while Level 2 involved reporting either the number seen by the avatar or whether it was to the avatar's left or right (Level 2). For both visual and spatial judgments, response times were longer for Level 2 tasks as the angular difference between the avatar and the participant increased, while response times remained unaffected by the angle in Level 1 tasks. This increase in response time when the participant's view was unaligned with the avatar's perspective is attributed to the mental rotation process, either rotating the scene or rotating one’s own reference frame to align with the avatar.

---

## Limitations of Spatial Assessment in Current Multimodal AI

Two primary limitations appear within AI spatial cognition literature: 1) linguistic reasoning can inflate performance on spatial benchmarks, and 2) benchmark scores can be hard to interpret when models perform poorly. For example, text-only GPT-4 achieves a score of 31.4, while multimodal GPT-4v achieves a score of 42.6 on the spatial understanding category of Meta's openEQA episodic memory task <d-cite key="majumdar2024openeqa"></d-cite>. The strong baseline score achieved by the text-only GPT-4 suggests that many "real-world" questions based on visual scenes can be deduced linguistically. Additionally, the limited improvement when moving from a blind LLM to a multimodal one suggests that vision models do not gain a significant understanding of space beyond what can be inferred through language.

Additionally, BLINK <d-cite key="fu2024blink"></d-cite>, a benchmark more specifically focused on visual perception capabilities, contains categories related to spatial cognition, such as relative depth and multi-view reasoning. On this benchmark, GPT-4v achieved an accuracy of 51.26\%, only 13.17\% higher than random guessing and 44.44\% lower than human performance. When benchmarks are highly focused on visuospatial tasks, the significant shortcomings of multimodal models suggest that further advancements are needed before these models can reliably perform in real-world scenarios. Even within specific categories, it is often difficult to determine {\it why} models fail on certain tasks while succeeding on others, as these failures cannot be easily linked to the absence of a particular cognitive process.

Here we apply established tasks in cognitive psychology that measure spatial cognition in a precise manner. By applying these tasks to AI systems, we gain not only improved measurement precision but also the ability to compare AI performance with human development, providing clear insights into model limitations and areas for improvement.

---

## Perspective Taking Benchmark

Leveraging the distinction between Level 1 and Level 2 perspective-taking <d-cite key="surtees2013similarities"></d-cite>, we propose a small perspective-taking benchmark that assesses multimodal model capabilities across three tasks: Level 1, Level 2 with spatial judgments, and Level 2 with visual judgments. Although human performance remains stable regardless of judgment type, we include this differentiation of Level 2 stimuli to examine potential egocentric biases that may arise in multimodal models when interpreting spatial relations compared to optical character recognition (OCR). This benchmark aims to address gaps in current AI spatial cognition measures by increasing process specificity, limiting language-based solutions, and offering straightforward comparisons to human cognition.

---

## Interactive Plots

You can add interative plots using plotly + iframes :framed_picture:

<div class="l-page">
  <iframe src="{{ '/assets/plotly/demo.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

The plot must be generated separately and saved into an HTML file.
To generate the plot that you see above, you can use the following code snippet:

{% highlight python %}
import pandas as pd
import plotly.express as px
df = pd.read_csv(
'https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv'
)
fig = px.density_mapbox(
df,
lat='Latitude',
lon='Longitude',
z='Magnitude',
radius=10,
center=dict(lat=0, lon=180),
zoom=0,
mapbox_style="stamen-terrain",
)
fig.show()
fig.write_html('assets/plotly/demo.html')
{% endhighlight %}

---

## Details boxes

Details boxes are collapsible boxes which hide additional information from the user. They can be added with the `details` liquid tag:

{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

---

## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

---

## Other Typography?

Emphasis, aka italics, with _asterisks_ (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or **underscores**.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
   ⋅⋅\* Unordered sub-list.
3. Actual numbers don't matter, just that it's a number
   ⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

- Unordered list can use asterisks

* Or minuses

- Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
http://www.example.com or <http://www.example.com> and sometimes
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style:
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

```python
s = "Python syntax highlighting"
print s
```

```
No language indicated, so no syntax highlighting.
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

| Markdown | Less      | Pretty     |
| -------- | --------- | ---------- |
| _Still_  | `renders` | **nicely** |
| 1        | 2         | 3          |

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can _put_ **Markdown** into a blockquote.

Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a _separate paragraph_.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the _same paragraph_.
