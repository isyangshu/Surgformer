close all; clear all;

phaseGroundTruths = {};
gt_root_folder = 'phase_annotations/';
for k = 41:80
    num = num2str(k);
    to_add = ['video-' num];
    video_name = [gt_root_folder to_add '.txt'];
    phaseGroundTruths = [phaseGroundTruths video_name];
end
% phaseGroundTruths = {'video41-phase.txt', ...
%     'video42-phase.txt'};
% phaseGroundTruths

phases = {'Preparation',  'CalotTriangleDissection', ...
    'ClippingCutting', 'GallbladderDissection',  'GallbladderPackaging', 'CleaningCoagulation', ...
    'GallbladderRetraction'};

fps = 1;
jaccard1 = zeros(7, 40);
prec1 = zeros(7, 40);
rec1 = zeros(7, 40);
acc1 = zeros(1, 40);
f11 = zeros(7, 40);


for i = 1:length(phaseGroundTruths)
    predroot = 'prediction/';
    phaseGroundTruth = phaseGroundTruths{i};
    predFile = [predroot 'video-' phaseGroundTruth(end-5:end-4) '.txt'];
    [gt] = ReadPhaseLabel(phaseGroundTruth);
    [pred] = ReadPhaseLabel(predFile);
    
    if(size(gt{1}, 1) ~= size(pred{1},1) || size(gt{2}, 1) ~= size(pred{2},1))
        error(['ERROR:' ground_truth_file '\nGround truth and prediction have different sizes']);
    end
    
    if(~isempty(find(gt{1} ~= pred{1})))
        error(['ERROR: ' ground_truth_file '\nThe frame index in ground truth and prediction is not equal']);
    end
    
    t = length(gt{2});
    for z=1:t
        gt{2}{z} = gt{2}{z}(1);
        pred{2}{z} = pred{2}{z}(1);
    end
    % reassigning the phase labels to numbers
    gtLabelID = [];
    predLabelID = [];
    for j = 1:7
        gtLabelID(find(strcmp(num2str(j-1), gt{2}))) = j;
        predLabelID(find(strcmp(num2str(j-1), pred{2}))) = j;
    end

    % compute jaccard index, precision, recall, and the accuracy
    [jaccard, prec, rec, acc, f1] = Evaluate(gtLabelID, predLabelID, fps);
    jaccard1(:, i) = jaccard;
    prec1(:, i) = prec;
    rec1(:, i) = rec;
    acc1(i) = acc;
    f11(:, i) = f1;
end

acc = acc1;
rec = rec1;
prec = prec1;
jaccard = jaccard1;
f1 = f11;

accPerVideo= acc;

% Compute means and stds
index = find(jaccard>100);
jaccard(index)=100;
meanJaccPerPhase = nanmean(jaccard, 2);
meanJaccPerVideo = nanmean(jaccard, 1);
meanJacc = mean(meanJaccPerPhase);
stdJacc = std(meanJaccPerPhase);
for h = 1:7
    jaccphase = jaccard(h,:);
    meanjaccphase(h) = nanmean(jaccphase);
    stdjaccphase(h) = nanstd(jaccphase);
end

index = find(f1>100);
f1(index)=100;
meanF1PerPhase = nanmean(f1, 2);
meanF1PerVideo = nanmean(f1, 1);
meanF1 = mean(meanF1PerPhase);
stdF1 = std(meanF1PerVideo);
for h = 1:7
    f1phase = f1(h,:);
    meanf1phase(h) = nanmean(f1phase);
    stdf1phase(h) = nanstd(f1phase);
end

index = find(prec>100);
prec(index)=100;
meanPrecPerPhase = nanmean(prec, 2);
meanPrecPerVideo = nanmean(prec, 1);
meanPrec = nanmean(meanPrecPerPhase);
stdPrec = nanstd(meanPrecPerPhase);
for h = 1:7
    precphase = prec(h,:);
    meanprecphase(h) = nanmean(precphase);
    stdprecphase(h) = nanstd(precphase);
end

index = find(rec>100);
rec(index)=100;
meanRecPerPhase = nanmean(rec, 2);
meanRecPerVideo = nanmean(rec, 1);
meanRec = mean(meanRecPerPhase);
stdRec = std(meanRecPerPhase);
for h = 1:7
    recphase = rec(h,:);
    meanrecphase(h) = nanmean(recphase);
    stdrecphase(h) = nanstd(recphase);
end


meanAcc = mean(acc);
stdAcc = std(acc);

% Display results
% fprintf('model is :%s\n', model_rootfolder);
disp('================================================');
disp([sprintf('%25s', 'Phase') '|' sprintf('%6s', 'Jacc') '|'...
    sprintf('%6s', 'Prec') '|' sprintf('%6s', 'Rec') '|']);
disp('================================================');
for iPhase = 1:length(phases)
    disp([sprintf('%25s', phases{iPhase}) '|' sprintf('%6.2f', meanJaccPerPhase(iPhase)) '|' ...
        sprintf('%6.2f', meanPrecPerPhase(iPhase)) '|' sprintf('%6.2f', meanRecPerPhase(iPhase)) '|']);
    disp('---------------------------------------------');
end
disp('================================================');

disp(['Mean jaccard: ' sprintf('%5.2f', meanJacc) '+-' sprintf('%5.2f', stdJacc)]);
disp(['Mean f1-score: ' sprintf('%5.2f', meanF1) '+-' sprintf('%5.2f', stdF1)]);
disp(['Mean accuracy: ' sprintf('%5.2f', meanAcc) '+-' sprintf('%5.2f', stdAcc)]);
disp(['Mean precision: ' sprintf('%5.2f', meanPrec) '+-' sprintf('%5.2f', stdPrec)]);
disp(['Mean recall: ' sprintf('%5.2f', meanRec) '+-' sprintf('%5.2f', stdRec)]);
