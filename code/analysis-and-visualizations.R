
library(rcompanion)
library(plyr) # for ddply
library(car) # for leveneTest, Anova
library(effectsize) # for cohens_d, eta_squared
library(afex) # for aov_ez
library(performance) # for check_normality, check_homogeneity
library(BayesFactor) # for ttestBF, anoveBF, posterior 
library(bayestestR) # for hdi
library(lme4) # for lmer
library(lmerTest)
library(reshape2) # for dcast
library(multpois)
library(ggplot2)
library(emmeans)
library(psych) #for getting phi coeffience in experiment 3


#Code documentation for analysis and visualizations in "Identifying Features Associated with Bias Against Stigmatized Groups in LLM
#Outputs and Guardrail Model Mitigation", AAAI 2026 submission

#### Experiment 1: Correlation between models and humans ####
#Experiment described in Section 6.1
#Results reported in Section 7.1

#load in data from humans' on social feature ratings and each LLMs' rating each social feature of 10 iterations
data <- read.csv('/data/results-from-pachankis-all.csv')

#correlations reported in Section 7.1 and illustrated in Figure 2
#data also presented in table 6 in appendix
cor.test(data$visibility.llama, data$visibility.participants) #not significant, pvalue = 0.0845
cor.test(data$course.llama, data$course.participants) # significant, pvalue =  0.01362
cor.test(data$disrupt.llama, data$disrupt.participants) #significant, pvalue = 1.054e-11 
cor.test(data$aesthetics.llama, data$aesthetics.participants) #significant, pvalue = 2.211e-14
cor.test(data$origin.llama, data$origin.participants) # significant, pvalue = 7.148e-08
cor.test(data$peril.llama, data$peril.participants) # significnat, pvalue = 3.163e-10

cor.test(data$visibility.mistral, data$visibility.participants) #significant, pvalue =  0.005876
cor.test(data$course.mistral, data$course.participants) # significant, pvalue = 0.02936
cor.test(data$disrupt.mistral, data$disrupt.participants) # significant, pvalue =  9.504e-09
cor.test(data$aesthetics.mistral, data$aesthetics.participants) # significant, pvalue = 1.098e-05
cor.test(data$origin.mistral, data$origin.participants) # not significant, pvalue = 0.5268
cor.test(data$peril.mistral, data$peril.participants) # significant,  pvalue = 0.001871

cor.test(data$visibility.granite, data$visibility.participants) #not significant, pvalue = 0.1008
cor.test(data$course.granite, data$course.participants) # significant, pvalue = 9.539e-05
cor.test(data$disrupt.granite, data$disrupt.participants) # significant, pvalue = 3.551e-06
cor.test(data$aesthetics.granite, data$aesthetics.participants) # significant, pvalue =  1.294e-11
cor.test(data$origin.granite, data$origin.participants) # not significant, pvalue = 0.7653
cor.test(data$peril.granite, data$peril.participants) # significant, pvalue =  2.406e-10


#Standard deviations (reported in Appendix, table 5)
sd(data$visibility.llama)
sd(data$visibility.mistral)
sd(data$visibility.granite)
sd(data$visibility.participants)

sd(data$course.llama)
sd(data$course.mistral)
sd(data$course.granite)
sd(data$course.participants)

sd(data$disrupt.llama)
sd(data$disrupt.mistral)
sd(data$disrupt.granite)
sd(data$disrupt.participants)

sd(data$aesthetics.llama)
sd(data$aesthetics.mistral)
sd(data$aesthetics.granite)
sd(data$aesthetics.participants)

sd(data$origin.llama)
sd(data$origin.mistral)
sd(data$origin.granite)
sd(data$origin.participants)

sd(data$peril.llama)
sd(data$peril.mistral)
sd(data$peril.granite)
sd(data$peril.participants)

#for easier graphing, we put the correlations manually into a new CSV and graphed correlations
data.fig2 = read.csv('/data/feature_correlations.csv')

#Figure 2
ggplot(data.fig2, aes(x = Dimensions, y = Correlation.to.Humans, fill = Model)) +
  geom_bar(stat = "identity", position = 'dodge') +
  ggtitle("Correlation of Ratings from LLMs to Humans per Model") +
  ylab("Correlation") +
  xlab("Dimension") +
  theme(legend.position = "top",
        plot.title = element_text(size = 30, hjust = 0.5, face="bold") ,
        axis.text.y= element_text(size=30),
        strip.text = element_text(size = 35),
        axis.title.y = element_text(size = 30),
        axis.text.x = element_text(size=30),
        axis.title.x = element_text(size=30),
        legend.spacing = unit(0.9, "cm"),
        legend.title = element_blank(),
        legend.text = element_text(size = 30)) +
  scale_x_discrete(guide = guide_axis(angle = 50))
ggsave("rq_1_correlation.svg")

#edit data for easier graphing for Figure 1 boxplot
feature <- rep(c(rep('concealability', times = 93), rep('persistent course', times = 93), rep('disruptiveness', times = 93), 
               rep('unappealing aesthetics', times = 93), rep('controllable origin', times = 93), rep('peril', times = 93)), 4)
model.or.human.participant <- c(rep('Participants',558), rep('Granite',558), rep('Llama',558), rep('Mistral',558))
avg.rating <- c(data$visibility.participants, data$course.participants, data$disrupt.participants, data$aesthetics.participants
                , data$origin.participants, data$peril.participants, 
                data$visibility.granite, data$course.granite, data$disrupt.granite, data$aesthetics.granite
                , data$origin.granite, data$peril.granite, 
                data$visibility.llama, data$course.llama, data$disrupt.llama, data$aesthetics.llama
                , data$origin.llama, data$peril.llama, 
                data$visibility.mistral, data$course.mistral, data$disrupt.mistral, data$aesthetics.mistral
                , data$origin.mistral, data$peril.mistral)
boxplot.data <- data.frame(feature = feature, rating = avg.rating, model.or.human = model.or.human.participant)
View(boxplot.data)

#Figure 1
boxplot.data$model.or.human <- factor(boxplot.data$model.or.human,
                                     levels = c('Participants','Granite', 'Llama', 'Mistral'),ordered = TRUE)

ggplot(boxplot.data, aes(x = model.or.human, y = rating, fill = model.or.human)) +
  geom_boxplot() +
  facet_grid(~feature) +
  scale_fill_brewer() +
  ylab("Ratings Across 93 Stigmas") +
  labs(title = "Average Rating of Social Features Across 93 Stigmas", fill = "Model or Human Participants:   ") + #this changes legend title
  theme(legend.position = "bottom",
        plot.title = element_text(size = 30, hjust = 0.5, face = "bold") , 
        # axis.line. text = element_text(size=15), 
        axis.text.y= element_text(size=25),
        axis.title.y = element_text(size = 25),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.x = element_blank(),
        legend.title = element_text(size = 25),
        legend.spacing = unit(0.9, "cm"),
        legend.text = element_text(size = 25),
        strip.text.x = element_text(size = 22)) +
  scale_x_discrete(guide = guide_axis(angle = 50))
ggsave("avg_dimension_ratings_test.svg")

#### Experiment 2: Effects on bias ####

#Experiment described in Section 6.2
#Results reported in Section 7.2

#load in data on if the model returned a biased answer for the socialstigmaQA prompts (across all 3 models)
socialStigmaQA_responses <- read.csv("/data/SSQA-results.csv") 

#Exploratory data analysis on amount of bias per cluster 
sum(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Threatening", ]$biased.answer) / nrow(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Threatening", ])
sum(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Sociodemographic", ]$biased.answer) / nrow(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Sociodemographic", ])
sum(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Awkward", ]$biased.answer) / nrow(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Awkward", ])
sum(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Innocuous Persistent", ]$biased.answer) / nrow(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Innocuous Persistent", ])
sum(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Unappealing Persistent", ]$biased.answer) / nrow(socialStigmaQA_responses[socialStigmaQA_responses$clusters == "Unappealing Persistent", ])

#Cluster type effect on bias
m_cluster = glmer(as.factor(biased.answer) ~ clusters + (1|model), data=socialStigmaQA_responses, family=binomial) 
Anova(m_cluster, type=3) #significant differences in clusters and in prompt type, but not in their interaction
emmeans(m_cluster, pairwise ~ clusters, adjust="holm")

#Prompt style effect on bias
m_promptstyle= glmer(as.factor(biased.answer) ~ prompt.style + (1|model), data=socialStigmaQA_responses, family=binomial) 
Anova(m_promptstyle, type=3)
emmeans(m_promptstyle, pairwise ~ prompt.style, adjust="holm")

#testing for interaction effect, not significant 
m_interaction= glmer(as.factor(biased.answer) ~ prompt.style * clusters + (1|model), data=socialStigmaQA_responses, family=binomial) 
Anova(m_interaction, type = 3)

#Feature ratings effect on biased outputs 
llama.percent.bias <- read.csv("/data/llama_bias_percentages.csv")
llama_column <- rep(c("llama"),times=nrow(llama.percent.bias))
llama.percent.bias$type <- llama_column

mistral.percent.bias <- read.csv("/data/mistral_bias_percentages.csv")
mistral_column <- rep(c("mistral"),times=nrow(mistral.percent.bias))
mistral.percent.bias$type <- mistral_column

granite.percent.bias <- read.csv("/data/granite_bias_percentages.csv")
granite_column <- rep(c("granite"),times=nrow(granite.percent.bias))
granite.percent.bias$type <- granite_column


mistral_less_info <- select(mistral,  percent.biased, percent.biased.post.guardian, Visibility , Persistent.Course, Disrupt ,
                            Unappealing.Aesthetics ,Controllable.Origin, Peril ,
                            visibility.human ,course.human	, disrupt.human ,	aesthetics.human , origin.human, peril.human, type)

llama_less_info <- select(llama,  percent.biased, percent.biased.post.guardian, Visibility , Persistent.Course, Disrupt , 
                          Unappealing.Aesthetics ,Controllable.Origin, Peril ,
                          visibility.human ,course.human	, disrupt.human ,	aesthetics.human , origin.human, peril.human, type)

granite_less_info <- select(granite,  percent.biased, percent.biased.post.guardian,Visibility , Persistent.Course, Disrupt ,
                            Unappealing.Aesthetics ,Controllable.Origin, Peril ,
                            visibility.human ,course.human	, disrupt.human ,	aesthetics.human , origin.human, peril.human, type)

combined_df <- rbind(mistral_less_info,llama_less_info,granite_less_info)
View(combined_df)

#Look at correlation between biased answer and LLM rating of Concealability, and Human ratings of Peril and Course (Section 7.2)
cor.test(combined_df$Visibility, combined_df$percent.biased) #0.5481954, p-value < 2.2e-16
cor.test(combined_df$peril.human, combined_df$percent.biased) #0.6503439 ,  p-value < 2.2e-16
cor.test(combined_df$course.human, combined_df$percent.biased) #-0.3740238, p-value = 1.081e-10

#run LMM
m = lmer(percent.biased ~  Visibility + Persistent.Course + Disrupt +
           Unappealing.Aesthetics + Controllable.Origin + Peril +
           visibility.human + course.human	+ disrupt.human +	aesthetics.human + origin.human + peril.human
         + (1|type), data=combined_df) # sphericity is N/A for LMMs
Anova(m, type=3, test.statistic="F")
r = residuals(m)
shapiro.test(r) # Shapiro-Wilk test passed , can continue using linear mixed model (rather than a generalized linear mixed model)

#### RESEARCH QUESTION 3: GUARDRAIL MODEL PERFORMANCE #### 

mistral.moderation.performance <- read.csv('/data/SSQA-performance-and-guardrail-mitigations/mistral_and_mistral_moderation.csv')
granite.guardian.performance <- read.csv('/data/SSQA-performance-and-guardrail-mitigations/granite_and_granite_guardian.csv')
llama.guard.performance <- read.csv('/data/SSQA-performance-and-guardrail-mitigations/llama_and_llama_guard.csv')
View(granite.guardian.performance)

#function to determine significance of difference between the original answer biases and post-mitigation answer biases
total_biased_outputs <- function(pre_or_post_column, table) {
  #1 is used when there was a biased answer
  sum(table[[pre_or_post_column]] == 1) / nrow(table)
}

#post guardrail LMER for seeing how features of stigma impact bias
m_post = lmer(percent.biased.post.guardian ~  Visibility + Persistent.Course + Disrupt +
                Unappealing.Aesthetics + Controllable.Origin + Peril +
                visibility.human + course.human	+ disrupt.human +	aesthetics.human + origin.human + peril.human
              + (1|type), data=combined_df)
Anova(m_post, type=3, test.statistic="F")
r_post = residuals(m_post) #they are all still the same post guardian! BUT the F statistics did decrease, but still significant 
shapiro.test(r_post)

#change in bias between llama
total_biased_outputs("original.answer.bias", llama.guard.performance) - total_biased_outputs("post.llama.answer.bias", llama.guard.performance)
#signifiance test for categorical bias vs no bias:
contingency_table.llama <- table( llama.guard.performance$original.answer.bias,  llama.guard.performance$post.llama.answer.bias)
mcnemar.test(contingency_table.llama) #we use mcnemar test for paired nominal variables
phi(contingency_table, digits = 3) #large effect size,  0.968

#change in bias between mistral
total_biased_outputs("original.answer.bias", mistral.moderation.performance) - total_biased_outputs("post.moderation.answer.bias", mistral.moderation.performance)
contingency_table.mistral <- table( mistral.moderation.performance$original.answer.bias,  mistral.moderation.performance$post.moderation.answer.bias)
mcnemar.test(contingency_table.mistral)
cohenG(contingency_table.mistral + 0.5) #Haldane-Anscombe Correction

#change in bias between granite for the default granite guardian performance (harm)
total_biased_outputs("original.answer.bias", granite.guardian.performance) - total_biased_outputs("post.guardian.answer.bias", granite.guardian.performance)
contingency_table.granite <- table( granite.guardian.performance$original.answer.bias,granite.guardian.performance$post.guardian.answer.bias)
mcnemar.test(contingency_table.granite)
oddsratio(contingency_table.granite + 0.5) #Haldane-Anscombe Correction

#change in bias between granite for the 'social base' granite guardian setting
total_biased_outputs("original.answer.bias", granite.guardian.performance) - total_biased_outputs("post.bias.guardian.answer.bias", granite.guardian.performance)
contingency_table.granite.bias <- table( granite.guardian.performance$original.answer.bias, granite.guardian.performance$post.bias.guardian.answer.bias)
mcnemar.test(contingency_table.granite.bias)
oddsratio(contingency_table.granite.bias + 0.5) #Haldane-Anscombe Correction

#visualize change in bias based on cluster type
#load in processed data for easier visualization
rq3_decrease_in_bias <- read.csv("/data/rq3_data_for_bias_change.csv")

#Figure 3
#data also presented in table 7 in appendix
ggplot(rq3_decrease_in_bias, aes(x = cluster, y = percent.bias.per.cluster, fill = type)) +
  geom_bar(stat = "identity", position = 'dodge') +
  scale_fill_manual(name = "Type ",values = c("blue", "red"), labels = c("Before Mitigation", "After Mitigation")) +
  facet_wrap(~model,  ncol=1, labeller = model_names) + 
  ggtitle("Decrease in Bias per Stigma Cluster") +
  ylab("Percent Bias Per Cluster") +
  xlab("Cluster") +
  theme(legend.position = "top",
        plot.title = element_text(size = 40, hjust = 0.5, face="bold") ,
        axis.text.y= element_text(size=30),
        strip.text = element_text(size = 35),
        axis.title.y = element_text(size = 30),
        axis.text.x = element_text(size=30),
        axis.title.x = element_text(size=30),
        legend.title = element_blank(),
        legend.spacing = unit(0.9, "cm"),
        legend.text = element_text(size = 30)) +
  scale_x_discrete(guide = guide_axis(angle = 50))
ggsave("rq3_cluster.svg")


#visualize change in bias based on prompt type
#load in processed data for easier visualizations
#data also presented in Table 8 in appendix
rq3_bias_per_prompt <- read.csv("/data/rq3_data_for_prompt_style.csv")

#Figure 4
ggplot(rq3_bias_per_prompt, aes(x = prompt.style, y = percent.bias.per.style, fill = type)) +
  geom_bar(stat = "identity", position = 'dodge') +
  scale_fill_manual(name = "Type ",values = c("blue", "red"), labels = c("Before Mitigation", "After Mitigation")) +
  facet_wrap(~model,  ncol=1, labeller = model_names) + 
  ggtitle("Decrease in Bias per Prompt Style") +
  ylab("Percent Bias Per Cluster") +
  xlab("Cluster") +
  theme(legend.position = "top",
        plot.title = element_text(size = 40, hjust = 0.5, face="bold") ,
        axis.text.y= element_text(size=30),
        strip.text = element_text(size = 35),
        axis.title.y = element_text(size = 30),
        axis.text.x = element_text(size=30),
        axis.title.x = element_text(size=30),
        legend.title = element_blank(),
        legend.spacing = unit(0.9, "cm"),
        legend.text = element_text(size = 30)) +
  scale_x_discrete(guide = guide_axis(angle = 50))
ggsave("rq3_promptstyle.svg")

#look at correlations for post-mitigation bias
cor.test(combined_df$Visibility, combined_df$percent.biased.post.guardian) #0.3665114 , p-value = 2.691e-10
cor.test(combined_df$peril.human, combined_df$percent.biased.post.guardian) #0.437638 , p-value = 1.756e-14
cor.test(combined_df$course.human, combined_df$percent.biased.post.guardian) #-0.3128568 , p-value = 9.458e-08

#analysis on intentionality - Gather % from Mistral and Llama categories

#percentage per each category for mistral moderation that was flagged
#table 9 in appendix
mistral.categories <- read.csv('/data/raw-guardrail-model-results/mistral-moderation-detection-results.csv')
attributes(mistral.categories)

total.flagged <- sum(mistral.categories$any_true_threshold == "True")

sum(mistral.categories$sexual_threshold == "True") / total.flagged
sum(mistral.categories$health_threshold== "True") / total.flagged
sum(mistral.categories$hate_and_discrimination_threshold== "True") / total.flagged
sum(mistral.categories$violence_and_threats_threshold == "True") / total.flagged
sum(mistral.categories$dangerous_and_criminal_content_threshold == "True") / total.flagged
sum(mistral.categories$selfharm_threshold== "True") / total.flagged
sum(mistral.categories$financial_threshold== "True") / total.flagged
sum(mistral.categories$law_threshold== "True") / total.flagged
sum(mistral.categories$pii_threshold== "True") / total.flagged

llama.categories <- read.csv('/data/SSQA-performance-and-guardrail-mitigations/llama_and_llama_guard.csv')
total.flagged.llama <- sum(llama.categories$llama.guard.unsafe.category != "none")
table(llama.categories$llama.guard.unsafe.category)

