% tagprop_path = '/home/xiaojie/Projects/kdgan/jingwei/model_based/tagprop/TagProp/';
% addpath(tagprop_path);
% tagmatrix = h5read('/home/xiaojie/Projects/data/yfcc100m/survey_data/yfcc8k/TextData/lemm_wordnet_freq_tags.h5', '/tagmatrix') > 0.5;
% tagmatrix = sparse(tagmatrix);
% NN = h5read('/home/xiaojie/Projects/data/yfcc100m/survey_data/yfcc8k/TagProp-data/yfcc8k/vgg-verydeep-16fc7relu,cosineknn,1000/nn_train.h5', '/NN');
% NN = NN(2:end, :);
% NN = double(NN);
% 
% X = NN;
% fprintf('%d', ndims(X)==2);
% fprintf('%d', isnumeric(X));
% fprintf('%d', min(X(:))>0);
% fprintf('%d', max(X(:))<=size(X,2));
% fprintf('%d', size(X,1)>0);

tagprop_path = '/home/xiaojie/Projects/kdgan/jingwei/model_based/tagprop/TagProp/';
addpath(tagprop_path);
load('/home/xiaojie/Projects/data/yfcc100m/survey_data/yfcc8k/TagProp-models/vgg-verydeep-16fc7relu,cosineknn,ranksigmoids,1000/model.mat');
tagmatrix = h5read('/home/xiaojie/Projects/data/yfcc100m/survey_data/yfcc8k/TextData/lemm_wordnet_freq_tags.h5', '/tagmatrix') > 0.5;
tagmatrix = sparse(tagmatrix);
NNT = h5read('/home/xiaojie/Projects/data/yfcc100m/survey_data/yfcc2k/TagProp-data/yfcc2k/yfcc8k/concepts.txt/vgg-verydeep-16fc7relu,cosineknn,1000/nn_test.h5', '/NNT');
NNT = double(NNT);


P = tagprop_predict(NNT,[],m)';
save('/home/xiaojie/Projects/data/yfcc100m/survey_data/yfcc2k/TagProp-Prediction/yfcc2k/yfcc8k/concepts.txt/tagprop/vgg-verydeep-16fc7relu,cosineknn,ranksigmoids,1000/prediction.mat', '-v7.3');
% exit;
