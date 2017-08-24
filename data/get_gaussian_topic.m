% get gaussian topic
% folder = 'images/';
% list = dir('images/*.mat');
% isfile = ~[list.isdir];
% filenames = {list(isfile).name};
% 
% allimages = [];
% for i=1:length(filenames)
%     S = load(fullfile(folder,filenames{i}),'images');
%     images = S.images;
%     allimages = [allimages;images];
% end
% ids = allimages(:,1);
% D = pdist2(u,allimages(:,2:end));
% [Y,I] = sort(D,2);
% k=10;
% num_topics = size(u,1);
% samples = zeros(num_topics,k);
% for i=1:num_topics
%     samples(i,:) = ids(I(i,1:k));
% end

imageFolder = '/media/bryan/diskd/Research/Project/connectioncnn/images/';
topicFolder = '/media/bryan/diskd/Research/Project/connectioncnn/topics/';
for i=1:num_topics
    outFolder = fullfile(topicFolder,num2str(i));
    if ~exist(outFolder,'dir')
        mkdir(outFolder);
    end
    for j=1:k
        imgPath = fullfile(imageFolder,strcat(num2str(samples(i,j)),'.jpg'));
        if exist(imgPath,'file')
            copyfile(imgPath,outFolder);
        end
    end
end