# Time-aware Predictions of Moments of Change in Longitudinal User Posts on Social Media
Code for the paper "Time-aware Predictions of Moments of Change in Longitudinal User Posts on Social Media" published at the 8th Workshop on Advanced Analytics and Learning on Temporal Data (AALTD) at ECML-PKDD 2023.

This code trains and evaluates different models to identify Moments of Change (MoCs) from longitudinally annotated textual datasets from Reddit and TalkLife. 


# Predicting Moments of Change
## Problem Statement
It is a multi-class classifcation problem, where we want to assess whether the $i$ th post $p_{i}$ from the $j$ th annotated timeline $T_{u,j}$, which is a sub-sequence of posts selected from the entire posting history $H_{u}$ of a user, $u$, contains which of the following labels 
$$p_{i} \in \lbrace S,E,O \rbrace$$ 
which correspond to "switch" (drastic mood change), "escalation" (gradual mood change), or "no change". Labels of $S$ and $E$ are defined as Moments of Change. 

When a given post is labelled as a MoC, a given range is provided where several surrounding posts in the provided span are also provided the same MoC label, and indicate how long this change in mood persists for. Thus this problem can also be approached as a span-based multi-class classification task. 

Each label was annotated with respect to all other posts in $T_{u,j}$ and annotations are made in isolation with respect to other timelines. 

# Structure of this Repository

* `\models\` contains the architecture of different models
* in `\utils\training\` and `\utils\evaluation\` you can train and evaluate the different models
* `\final_experiments\` contains entire pipelines, which train and evaluate the proposed models and report the final set of results which would be included in the paper's submission.
* `\utils\data_handler\` contains several useful pieces of code for handling data - in particular loading the TalkLife and Reddit datasets.
