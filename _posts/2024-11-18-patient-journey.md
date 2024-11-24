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
  - name: Representation and Embeddings
  - name: Clustering
  - name: "Case Study: Lung Cancer Dataset"
  - name: Results
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

Properly mapping out Patient Journeys in an healthcare system involves the analysis of both structured and unstructured dataset.  Longitudinal data allows for evaluating the effectiveness of treatments over time but because this data is time-series in nature, it is difficult to apply standard clustering and prediction algorithms. This work explores a novel approach in mapping out Patient Journeys by combining Large Language Model (LLM) embeddings, K-Means clustering, and LLM based summarization. Embeddings are used to encode and project timeseries treatment regimes into a constant dimension vector, allowing K-Means to group journeys with similar patterns. Subsequently, an LLM summarizes the characteristics of each patient journey cluster, providing explainations on why patients are grouped together.  For instance, identifying clusters of patients with high readmission rates can help proactively address underlying issues. The resulting understanding of patient experiences can inform the development of targeted interventions and improve healthcare delivery<d-cite key="advancing,GENTRY2023275"></d-cite>.
Patient Journey helps in uncovering patient response variability, factors influencing variability, such as genetic markers, comorbidities, or socio-demographic factors. Finally, these journeys can be critical in the context of clinical trials for identifying eligible patients for trials and tracking participant progress during trials.

---

## Representation and Embeddings

Our method transforms raw patient data into informative embeddings that capture the relationships between treatments, diagnostic events, genomic sequencing, and patient responses. The first step in our methodology involves extracting data into a structured format that includes event type, event description, and event date. The event type categorizes whether the event is therapeutic (e.g., drug treatment), diagnostic, or pertains to a change in the patient's condition. The event description provides detailed information about the event, such as drug names and dosages or a textual description of the diagnosis. The event date is used to chronologically order these events.

Inside the event description, there is a large amount of irrelevant information (dosages, admission route, etc.) included in the health record. To overcome this we utilize tokenization to only extract the key information for diagnosis and/or treatment from each respective event<d-cite key="noauthor_pdf_nodate"></d-cite>.

For each patient $$i$$, let the set of event times be $$ {t_1, t_2, \dots, t_n} $$. The first event time $$t_{\text{min}} $$ is calculated as:

$$ t_{\text{min}}^i = \min(t_1, t_2, \dots, t_n) $$

This gives the earliest recorded event or treatment date for each patient indexed by $$ i $$. For each event $$ t_j $$ compute the treatment day as the difference between the event time and the first event date:

$$\text{treatment day}^i_j = t_j - t_{\text{min}}^i$$

This normalizes all event times to a relative day count starting from the first treatment. Then for each treatment day the drugs administered are aggregated into a summary string:

$$\text{summary}^i_j = \text{join(drugs administered on day } t_j)$$

This concatenates the drugs administered on a given day into a single record. The data is sorted by patient and event time to ensure a chronological order of treatments:

$$\text{sorted}(t_j^i, \text{ treatment day}^i_j)$$.

For example, given the following set of treatment events:

```
[
    {
        "Event Type": "Medication",
        "Event Attribute": "Chemotherapy",
        "Event Description": "pemetrexed",
        "Event Time": "2016-06-01T00:00:00",
    },
    {
        "Event Type": "Medication",
        "Event Attribute": "Chemotherapy",
        "Event Description": "pemetrexed",
        "Event Time": "2016-06-01T00:00:00",
    },
    {
        "Event Type": "Medication",
        "Event Attribute": "Chemotherapy",
        "Event Description": "cisplatin",
        "Event Time": "2016-03-01T00:00:00",
    },
]
```

This would be transformed into the following text string: `"pemetrexed administered on day 0, cisplatin administered on day 90"`.

These text strings were then used as input into a LLM to create a set of vector embeddings which can be used as a numerical representation for further analysis. These embeddings capture the semantic and syntactic nuances of the patient's journey, allowing us to compare and analyze different patient experience.

---

## Clustering

Once we've transformed our patient journeys into numerical representations, we can analyze them using clustering techniques to categorize patients and discover patterns within the data. For this post we will focus on the K-Means algorithm using $$L2$$ distance. K-Means clustering, which aims to partition the data into $$k$$ clusters. The objective is to minimize the within-cluster sum of squares (WCSS):

$$
\underset{S_1, ..., S_k}{\min} \sum_{i=1}^k \sum_{x \in S_i} ||x - \mu_i||^2
$$

where $$S_i$$ is the i-th cluster, $$μ_i$$ is the mean (centroid) of $$S_i$$, and ∣∣⋅∣∣. K-Means iteratively assigns data points to the nearest cluster centroid and then recalculates the centroids. This process continues until convergence or a maximum number of iterations is reached.

Cluster 0 are patients  with squamous cell carcinoma and malignant neoplasms with  references to genetic mutation panels and targeted therapies like osimertinib.

Cluster 1 are patients with secondary malignant neoplasms, particularly of the left lung, and overlapping sites of bronchus and lung and treatments include osimertinib and erlotinib.

Cluster 2 are patients with a progression of adenocarcinoma in the right lung, frequently transitioning from stage 3 to stage 4 with treatments of osmertinib.

Cluster 3 are patients metastatic lung cancer with secondary malignancies, such as those affecting the bone or liver and treatments with osmertinib and erlotinib.

Cluster 4 are patients with  malignant neoplasms of the upper lobe of the left lung and  metastatic of colorectal cancer treatments involving osimertinib and cases of metastatic colorectal cancer alongside primary lung cancer.

---

## Case Study: Analysis of Lung Cancer Data

To demonstrate our algorithms, we use a de-identified dataset of cancer patients<d-cite key="naik2024applying"></d-cite>.
For this demo we wish to focus only on patients that have been diagnosed with Lung Cancer.
Thus we select only patients that have an ICD-10 diagnosis that starts with C34. 
This gives us a group of 441 patients, 198 of which have medication info and 272 with genomic panels.

We then take the resulting treatment data for each patient and create the embedding vector for each patient using GPT-4<d-cite key="openai_gpt-4_2024"></d-cite>. With these embeddings, patients are clustered into 5 groups. The following plot, shows Sankey diagrams that illustrate the flow of treatment for each group.

<!-- <div class="l-page">
  <iframe src="{{ '/assets/plotly/demo.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div> -->
<div class="l-page">
  <iframe src="{{ '/assets/plotly/diagnostic.html' | relative_url }}" frameborder='0' scrolling='yes' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

---

## Conclusion

This post shows the potential of LLMs for the representation and then analyse of patient journeys. We demonstrate how LLM embeddings can be combined with simple ML algorithms to gain insights from longitudinal patient data.

To further enhance our patient journey analysis, we plan to expand our data sources to include comprehensive mutation information, demographic data, and unstructured data from histopathology and genomic reports. We will further expirement with other LLMs for embedding generation to improve the quality of patient representations. Experiments with additional clustering algorithms can also lead to additional insites into the data. By employing this approach, we plan to uncover deeper patterns in patient journeys and enable a new avenue for performing population level analysis of cancer data.
