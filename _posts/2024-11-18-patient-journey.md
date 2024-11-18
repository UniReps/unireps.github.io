---
layout: distill
title: Representing Patient Journeys in Healthcare with Large Language Models
tags: patient-journey llm healthcare representation-learning
giscus_comments: true
date: 2024-11-18
featured: true

authors:
  - name: Narmada Naik
    url: "https://www.datma.com/"
    affiliations:
      name: datma
  - name: Kevin Matlock
    url: "https://www.datma.com/"
    affiliations:
      name: datma

bibliography: 2024-11-18-patient-journey.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: "Case Study: Non-Small Cell Lung Cancer Dataset"
  - name: Representation and Embeddings
  - name: Clustering
  - name: Results
  - name: Federated Patient Journey
  - name: Conclusion

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

Properly mapping out Patient Journeys in an healthcare system involves the analysis of both structured and unstructured dataset. In addition, this data is time-series in nature, making it difficult to apply standard clustering and prediction algorithms. This work explores a novel approach in mapping out Patient Journeys by combining Large Language Model (LLM) embeddings, K-Means clustering, and LLM based summarization. Embeddings are used to encode and project timeseries treatment regimes into a constant dimension vector, allowing K-Means to group journeys with similar patterns. Subsequently, an LLM summarizes the characteristics of each patient journey cluster, providing explainations on why patients are grouped together.  For instance, identifying clusters of patients with high readmission rates can help proactively address underlying issues. The resulting understanding of patient experiences can inform the development of targeted interventions and improve healthcare delivery.


---

## Case Study: Non-Small Cell Lung Cancer Dataset

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

---

## Representation and Embeddings

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

---

## Clustering

Syntax highlighting is provided within `<d-code>` tags.
An example of inline code snippets: `<d-code language="html">let x = 10;</d-code>`.
For larger blocks of code, add a `block` attribute:

<d-code block language="javascript">
  var x = 25;
  function(x) {
    return x * x;
  }
</d-code>

**❗️ Note:** `<d-code>` blocks do not look good in the dark mode. ❗️

You can always use the default code-highlight using the `highlight` liquid tag:

{% highlight javascript %}
var x = 25;
function(x) {
return x \* x;
}
{% endhighlight %}

---

## Results

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

## Federated Patient Journey

Details boxes are collapsible boxes which hide additional information from the user. They can be added with the `details` liquid tag:

{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

---

## Conclusion

I am a conclusion