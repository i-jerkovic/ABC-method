Adjusted binary classification (ABC) model in forensic science: an
example on sex classification from handprint dimensions
================
13/12/2020

1.  Reading packages

<!-- end list -->

``` r
library(readxl)
library(dplyr)
library(caret)
library(expss)
library(tidyverse)
```

2.  Loading new “round” function

<!-- end list -->

``` r
round2 = function(x, n) {
  posneg = sign(x)
  z = abs(x)*10^n
  z = z + 0.5
  z = trunc(z)
  z = z/10^n
  z*posneg
}
```

3.  Loading training and testing example data form github.
    Alternatively, the user can select other data and name traing set -
    “training” , and testing set - “testing” without changing the code
    below.

<!-- end list -->

``` r
load(url("https://github.com/i-jerkovic/ABC-method/blob/master/Rdata/training.RData?raw=true"))

load(url("https://github.com/i-jerkovic/ABC-method/blob/master/Rdata/testing.RData?raw=true"))
```

4.  Setting seed for cross-validation function.

<!-- end list -->

``` r
RNGkind(sample.kind = "Rejection")
set.seed(12345)
```

5.  Defining validation method and performance parameters.

In this case, we used leave group out or Monte Carlo cross-validation
algorithm (method=“LGOCV”).  
We used 70% of the data for the calibration set (p = 0.7), while 30%
were used for validation.  
This procedure was iterated 50 times (n = 50) for each model.

``` r
train_control <- trainControl(method="LGOCV", p = 0.7, n = 50, classProbs = TRUE, savePredictions = "final", summaryFunction = twoClassSummary)
```

6.  In this example, we selected the variable “HB” from the training
    sample.  
    Linear discriminant analysis (method=“lda”) was selected as a
    classification method.  
    However, other classification methods can also be easily employed.  
    See ? train\_model\_list

<!-- end list -->

``` r
model <- train(sex ~  HB,  data=training, trControl=train_control, method="lda", metric = "ROC")
```

This data shows the results of cross-validation.  
The variable pred indicates sex predicted by classification model.  
The variable obs indicates observed data, i.e., initial data on sex.  
X1 shows the posterior probability of an individual being male.  
X2 shows the posterior probability of an individual being female.  
Here we show only first 6 rows from 1600.

``` r
head(model$pred)
```

    ##   parameter pred obs        X1         X2 rowIndex   Resample
    ## 1      none   X1  X1 0.9832831 0.01671686        4 Resample01
    ## 2      none   X1  X1 0.7495943 0.25040570        6 Resample01
    ## 3      none   X1  X1 0.8910254 0.10897456       11 Resample01
    ## 4      none   X1  X1 0.7142752 0.28572481       12 Resample01
    ## 5      none   X1  X1 0.7914001 0.20859987       15 Resample01
    ## 6      none   X1  X1 0.9306202 0.06937980       17 Resample01

7.  Classification performance metrics of the model without adjustment
    was performed by comparing the predicted and observed data on sex.  
    Note: Classification performance metrics were calculated as the
    average of all iterations (n = 1600).

<!-- end list -->

``` r
cvmatrix <- confusionMatrix(model$pred$pred, model$pred$obs)
cvmatrix
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  X1  X2
    ##         X1 736  56
    ##         X2  64 744
    ##                                          
    ##                Accuracy : 0.925          
    ##                  95% CI : (0.911, 0.9374)
    ##     No Information Rate : 0.5            
    ##     P-Value [Acc > NIR] : <2e-16         
    ##                                          
    ##                   Kappa : 0.85           
    ##                                          
    ##  Mcnemar's Test P-Value : 0.5228         
    ##                                          
    ##             Sensitivity : 0.9200         
    ##             Specificity : 0.9300         
    ##          Pos Pred Value : 0.9293         
    ##          Neg Pred Value : 0.9208         
    ##              Prevalence : 0.5000         
    ##          Detection Rate : 0.4600         
    ##    Detection Prevalence : 0.4950         
    ##       Balanced Accuracy : 0.9250         
    ##                                          
    ##        'Positive' Class : X1             
    ## 

8.  From the same model, we extracted posterior probabilities separately
    for males and females.  
    For male individuals, we extracted posterior probabilities of being
    male. -\> mprob  
    For female individuals, we extracted posterior probabilities of
    being female. -\> fprob  
    To manipulate data more easily, we rounded posterior probabilities
    to three decimals.

<!-- end list -->

``` r
mprob <- round2(model$pred$X1, 3)
fprob <- round2(model$pred$X2, 3)
```

``` r
head(mprob)
```

    ## [1] 0.983 0.750 0.891 0.714 0.791 0.931

``` r
head(fprob)
```

    ## [1] 0.017 0.250 0.109 0.286 0.209 0.069

9.  We constructed a new data frame that includes observed sex, male and
    female posterior probabilities.

<!-- end list -->

``` r
predictions <- data.frame(sex = model$pred$obs, mprob, fprob)
```

``` r
head(predictions)
```

    ##   sex mprob fprob
    ## 1  X1 0.983 0.017
    ## 2  X1 0.750 0.250
    ## 3  X1 0.891 0.109
    ## 4  X1 0.714 0.286
    ## 5  X1 0.791 0.209
    ## 6  X1 0.931 0.069

We extracted information of sex predicted by the model.

``` r
predsex <- model$pred$pred

head(predsex)
```

    ## [1] X1 X1 X1 X1 X1 X1
    ## Levels: X1 X2

Finally, we added that information in ‘predictions’.

``` r
predictions <- mutate(predictions, predsex)

head(predictions)
```

    ##   sex mprob fprob predsex
    ## 1  X1 0.983 0.017      X1
    ## 2  X1 0.750 0.250      X1
    ## 3  X1 0.891 0.109      X1
    ## 4  X1 0.714 0.286      X1
    ## 5  X1 0.791 0.209      X1
    ## 6  X1 0.931 0.069      X1

We compared observed and predicted data on sex and constructed a new
variable that indicates if sex was accurately classified. This variable
was added to the ‘predictions’ data.

``` r
corrClass <- predictions$sex == predictions$predsex

predictions <- mutate(predictions, corrClass)

head(predictions)
```

    ##   sex mprob fprob predsex corrClass
    ## 1  X1 0.983 0.017      X1      TRUE
    ## 2  X1 0.750 0.250      X1      TRUE
    ## 3  X1 0.891 0.109      X1      TRUE
    ## 4  X1 0.714 0.286      X1      TRUE
    ## 5  X1 0.791 0.209      X1      TRUE
    ## 6  X1 0.931 0.069      X1      TRUE

10. The table ‘predictions’ is divided on males and females.

<!-- end list -->

``` r
predictions_m <- predictions %>% filter(predsex == "X1")
predictions_f <- predictions %>% filter(predsex == "X2")
```

ABC adjustment - for males

11. We created two empty vectors:

acc\_m - which will contain data on accuracies when different posterior
probabilities are used.  
pp\_m - which will contain data on posterior probabilities

``` r
acc_m <- vector()
pp_m <- vector ()
```

12. We created a loop that examines the accuracy of the model when we
    classify only specimens with posterior probabilities higher then
    chosen threshold. In this case, we examined thresholds from 0.5 to 1
    with increment of 0.001.

<!-- end list -->

``` r
for(i in 500:1000) {
  j <- i/1000
  acc_m[i] <- with(predictions_m, mean_if(ge(j), mprob, data = corrClass))*100
  pp_m[i] <- j
}
```

Results are provided in the following plot.

``` r
acc_pp_m <- data.frame(accuracy = acc_m, probability = pp_m)

acc_pp_m %>% ggplot(aes(probability, accuracy)) + geom_line() + geom_hline(yintercept = 95, color='red')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

13. From the previous plot, we can see that at the certain value of
    posterior probability, accuracy reaches 95% and does not fall below
    this value.  
    So, in the next step, we calculated the minimum value of posterior
    probability that provides the accuracy of at least 95%.  

For this reason, we extract all posterior probabilities that provide
accuracies lower than 95%.

``` r
lessm <- filter(acc_pp_m, accuracy < 95)
head(lessm)
```

    ##   accuracy probability
    ## 1 92.92929       0.500
    ## 2 92.92929       0.501
    ## 3 92.92035       0.502
    ## 4 93.03797       0.503
    ## 5 93.03797       0.504
    ## 6 93.03797       0.505

``` r
tail(lessm)
```

    ##     accuracy probability
    ## 103 94.87633       0.876
    ## 104 94.87633       0.877
    ## 105 94.87633       0.878
    ## 106 94.85816       0.879
    ## 107 94.84902       0.880
    ## 108 94.96403       0.884

14. In the next step, we look for a maximum value of probability that
    provides accuracy lower than 95%.

<!-- end list -->

``` r
maxppm <- ifelse(nrow(lessm)==0, 0.500, max(lessm$probability))
```

Therefore, the probability threshold for males is:

``` r
maxppm
```

    ## [1] 0.884

``` r
acc_pp_m %>% ggplot(aes(probability, accuracy)) + geom_line() + geom_hline(yintercept = 95, color='red') + geom_vline(xintercept = maxppm, color ='blue')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

15. Now, we compute the proportion of specimens with probability greater
    than the threshold. First, we create an empty vector:  

avlbm - contains the number of specimens that can be classified if a
certain probability threshold is applied.  

``` r
avlbm <- vector()
```

16. We created the same loop that will calculate a number of classified
    individuals for probability thresholds from 0.5 to 1 with an
    increment of 0.001.

<!-- end list -->

``` r
for(i in 500:1000) {
  j <- i/1000
  avlbm[i] <- with(predictions_m, count_if(ge(j), mprob, data = predsex))
}
```

The number of classified specimens is furtherly converted to the
proportion.

``` r
avlbm <- avlbm / length(predictions_m$sex) * 100
```

The results are provided in the following plot.

``` r
avlb_pp_m <- data.frame(availability = avlbm, probability = pp_m)

avlb_pp_m %>% ggplot(aes(probability, availability)) + geom_line() + geom_vline(xintercept = maxppm, color='red')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

17. We calculate the proportion of classified specimens for a threshold
    that provides 95% accuracy.

<!-- end list -->

``` r
maxm_perc <- avlb_pp_m[which(avlb_pp_m$probability == maxppm),1]
maxm_perc
```

    ## [1] 70.20202

The influence of raising probability to the proportion of classified
specimens is shown in the following plot.

``` r
avlb_pp_m %>% ggplot(aes(probability, availability)) + geom_line() + geom_vline(xintercept = maxppm, color='red')+ geom_hline(yintercept = maxm_perc, color='blue')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

18. Now, we repeat the same procedure for females.  
    We create empty vectors:  
    acc\_f - which will contain data on accuracies when different
    posterior probabilities are used  
    pp\_f - which will contain data on posterior probabilities  

<!-- end list -->

``` r
acc_f <- vector()
pp_f <- vector ()
```

19. We again create a loop that examines the accuracy of the model when
    we classify only specimens with a posterior probability higher then
    chosen threshold. In this case, we examined thresholds from 0.5 to 1
    with an increment of 0.001.

<!-- end list -->

``` r
acc_f <- vector()
pp_f <- vector ()

for(i in 500:1000) {
  j <- i/1000
  acc_f[i] <- with(predictions_f, mean_if(ge(j), fprob, data = corrClass))*100
  pp_f[i] <- j
}
```

The results are provided in the following plot.

``` r
acc_pp_f <- data.frame(accuracy = acc_f, probability = pp_f)
acc_pp_f %>% ggplot(aes(probability, accuracy)) + geom_line() + geom_hline(yintercept = 95, color='red')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-33-1.png)<!-- -->

20. From the previous plot, we can see that at certain value of
    posterior probability, accuracy reaches 95% and does not fall below
    this value.  
    So, in the next step, we calculated the minimum value of posterior
    probability that provides the accuracy of at least 95%.  

For this reason, we extract all posterior probabilities that provide
accuracies lower than 95%.

``` r
lessf <- filter(acc_pp_f, accuracy < 95)
head(lessf)
```

    ##   accuracy probability
    ## 1 92.07921       0.500
    ## 2 92.07921       0.501
    ## 3 92.07921       0.502
    ## 4 92.07921       0.503
    ## 5 92.06939       0.504
    ## 6 92.06939       0.505

``` r
tail(lessf)
```

    ##     accuracy probability
    ## 181 94.23929       0.680
    ## 182 94.23077       0.681
    ## 183 94.65082       0.682
    ## 184 94.93294       0.683
    ## 185 94.93294       0.684
    ## 186 94.93294       0.685

21. In the next step, we look for a maximum probability value that
    provides accuracy lower than 95%.

<!-- end list -->

``` r
maxppf <- ifelse(nrow(lessf)==0, 0.500, max(lessf$probability))
```

Therefore, the probability threshold for the females is:

``` r
maxppf
```

    ## [1] 0.685

``` r
acc_pp_f %>% ggplot(aes(probability, accuracy)) + geom_line() + geom_hline(yintercept = 95, color='red') + geom_vline(xintercept = maxppf, color ='blue')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-37-1.png)<!-- -->

22. Now, we compute the proportion of specimens with probability greater
    than the threshold. First, we create an empty vector:  

avlbf contains the number of specimens that can be classified if a
certain probability threshold is applied.  

``` r
avlbf <- vector()
```

23. We created the same loop that will calculate the number of
    classified individuals for probability thresholds from 0.5 to 1 with
    an increment of 0.001.

<!-- end list -->

``` r
for(i in 500:1000) {
  j <- i/1000
  avlbf[i] <- with(predictions_f, count_if(ge(j), fprob, data = predsex))
}
```

The number of classified specimens is furtherly converted to the
proportion.

``` r
avlbf <- avlbf / length(predictions_f$sex) * 100
```

The results are provided in the following plot.

``` r
avlb_pp_f <- data.frame(availability = avlbf, probability = pp_f)

avlb_pp_f %>% ggplot(aes(probability, availability)) + geom_line() + geom_vline(xintercept = maxppf, color='red')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-41-1.png)<!-- -->

23. We calculate the proportion of classified specimens for a threshold
    that provides 95% accuracy.

<!-- end list -->

``` r
maxf_perc <- avlb_pp_f[which(avlb_pp_f$probability == maxppf),1]
maxf_perc
```

    ## [1] 83.04455

``` r
avlb_pp_f %>% ggplot(aes(probability, availability)) + geom_line() + geom_vline(xintercept = maxppf, color='red')+ geom_hline(yintercept = maxf_perc, color='blue')
```

![](abc_code_example_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

24. Now, we can test cross-validated model performance using calculated
    thresholds for males and females. So, we will estimate sex only for
    those specimens that meet the posterior probability criteria.

<!-- end list -->

``` r
CVtestdata <- predictions %>% filter(mprob > maxppm | fprob > maxppf)
```

25. Classification performance metrics of the model with ABC adjustment
    was performed by comparing the predicted and observed data on sex.  
    Note: Classification performance metrics were calculated as the
    average of all iterations (n = 1600).

<!-- end list -->

``` r
cvABCmatrix <- confusionMatrix(CVtestdata$predsex, CVtestdata$sex)
cvABCmatrix
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  X1  X2
    ##         X1 527  26
    ##         X2  33 636
    ##                                          
    ##                Accuracy : 0.9517         
    ##                  95% CI : (0.9382, 0.963)
    ##     No Information Rate : 0.5417         
    ##     P-Value [Acc > NIR] : <2e-16         
    ##                                          
    ##                   Kappa : 0.9027         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.4347         
    ##                                          
    ##             Sensitivity : 0.9411         
    ##             Specificity : 0.9607         
    ##          Pos Pred Value : 0.9530         
    ##          Neg Pred Value : 0.9507         
    ##              Prevalence : 0.4583         
    ##          Detection Rate : 0.4313         
    ##    Detection Prevalence : 0.4525         
    ##       Balanced Accuracy : 0.9509         
    ##                                          
    ##        'Positive' Class : X1             
    ## 

26. The model’s performance is furtherly checked on the test data set.

First, we predict data using the unadjusted model and new data set -
testing.

``` r
predicted <- predict(model, newdata = testing, type = "prob")
head(predicted)
```

    ##          X1           X2
    ## 1 0.5342203 0.4657797252
    ## 2 0.9923884 0.0076115584
    ## 3 0.2531310 0.7468690467
    ## 4 0.9752852 0.0247147612
    ## 5 0.9998318 0.0001681582
    ## 6 0.9840423 0.0159576746

27. Then, we add those predictions to the test set which contains
    observed sex data.  
    Posterior probabilities are rounded to 3 decimals.

<!-- end list -->

``` r
testing <- mutate(testing, predm = predicted$X1, predf = predicted$X2)
testing$predm <- round2(testing$predm, 3)
testing$predf <- round2(testing$predf, 3)
```

28. If the estimated probability is larger for males, the specimen is
    classified as male (X1). Otherwise, the specimen is classified as
    female.

\`

``` r
testing <- mutate(testing, ac = ifelse(predm>predf, "X1", "X2"))
```

``` r
testmatrix <- confusionMatrix(as.factor(testing$ac), testing$sex)
testmatrix
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction X1 X2
    ##         X1 23  2
    ##         X2  1 22
    ##                                          
    ##                Accuracy : 0.9375         
    ##                  95% CI : (0.828, 0.9869)
    ##     No Information Rate : 0.5            
    ##     P-Value [Acc > NIR] : 6.563e-11      
    ##                                          
    ##                   Kappa : 0.875          
    ##                                          
    ##  Mcnemar's Test P-Value : 1              
    ##                                          
    ##             Sensitivity : 0.9583         
    ##             Specificity : 0.9167         
    ##          Pos Pred Value : 0.9200         
    ##          Neg Pred Value : 0.9565         
    ##              Prevalence : 0.5000         
    ##          Detection Rate : 0.4792         
    ##    Detection Prevalence : 0.5208         
    ##       Balanced Accuracy : 0.9375         
    ##                                          
    ##        'Positive' Class : X1             
    ## 

29. Now, we consider only specimens that have posterior probabilities
    larger than previously calculated thresholds.

<!-- end list -->

``` r
testing2<- testing %>% filter(predm > maxppm | predf > maxppf)
testing2$ac <- factor(testing2$ac)
```

30. Lastly, the confusion matrix is created for testing sample.

<!-- end list -->

``` r
testmatrixABC <- confusionMatrix(testing2$ac, testing2$sex)
testmatrixABC
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction X1 X2
    ##         X1 15  0
    ##         X2  1 21
    ##                                           
    ##                Accuracy : 0.973           
    ##                  95% CI : (0.8584, 0.9993)
    ##     No Information Rate : 0.5676          
    ##     P-Value [Acc > NIR] : 2.311e-08       
    ##                                           
    ##                   Kappa : 0.9445          
    ##                                           
    ##  Mcnemar's Test P-Value : 1               
    ##                                           
    ##             Sensitivity : 0.9375          
    ##             Specificity : 1.0000          
    ##          Pos Pred Value : 1.0000          
    ##          Neg Pred Value : 0.9545          
    ##              Prevalence : 0.4324          
    ##          Detection Rate : 0.4054          
    ##    Detection Prevalence : 0.4054          
    ##       Balanced Accuracy : 0.9688          
    ##                                           
    ##        'Positive' Class : X1              
    ## 

31. In the testing sample, we also calculate the proportion of
    classified individuals.

<!-- end list -->

``` r
test_males_proportion <- (testmatrixABC$table[1]+testmatrixABC$table[2])/sum(testing$sex=="X1")*100
test_males_proportion
```

    ## [1] 66.66667

``` r
test_females_proportion <- (testmatrixABC$table[3]+testmatrixABC$table[4])/sum(testing$sex=="X2")*100
test_females_proportion
```

    ## [1] 87.5

32. Summary of the unadjusted and adjusted model.

<!-- end list -->

``` r
summaryCV <- c(cvmatrix$overall[1], cvmatrix$byClass[1], cvmatrix$byClass[2], cvmatrix$byClass[3], cvmatrix$byClass[4], classified_m = 100, classified_f = 100, cut_off_m = 0.5, cut_off_f = 0.5)
               
summaryCVABC <- c(cvABCmatrix$overall[1], cvABCmatrix$byClass[1], cvABCmatrix$byClass[2], cvABCmatrix$byClass[3], cvABCmatrix$byClass[4], classified_m = maxm_perc, classified_f=maxf_perc, cut_off_m = maxppm, cut_off_f = maxppf)

summaryTEST <-  c(testmatrix$overall[1], testmatrix$byClass[1], testmatrix$byClass[2], testmatrix$byClass[3], testmatrix$byClass[4], classified_m = 100, classified_f=100, cut_off_m = 0.5, cut_off_f = 0.5)

summaryTESTABC <- c(testmatrixABC$overall[1], testmatrixABC$byClass[1], testmatrixABC$byClass[2], testmatrixABC$byClass[3], testmatrixABC$byClass[4], classified_m = test_males_proportion, classified_f=test_females_proportion, cut_off_m = maxppm, cut_off_f = maxppf)

round2(cbind(LGOCV = summaryCV, LGOCV_ABC = summaryCVABC, TEST = summaryTEST, TESTABC=summaryTESTABC), 3)
```

    ##                  LGOCV LGOCV_ABC    TEST TESTABC
    ## Accuracy         0.925     0.952   0.938   0.973
    ## Sensitivity      0.920     0.941   0.958   0.938
    ## Specificity      0.930     0.961   0.917   1.000
    ## Pos Pred Value   0.929     0.953   0.920   1.000
    ## Neg Pred Value   0.921     0.951   0.957   0.955
    ## classified_m   100.000    70.202 100.000  66.667
    ## classified_f   100.000    83.045 100.000  87.500
    ## cut_off_m        0.500     0.884   0.500   0.884
    ## cut_off_f        0.500     0.685   0.500   0.685
