filelist = dir('images/*.mat');
filenames={filelist.name};
for i=1:length(filenames)
    [path,name,ext] = fileparts(filenames{i});
    user = str2num(name);
    if ismember(user,sample_users)
        fprintf('%d exists in sample_users\n',user);
    else
        fprintf('%d does not exist in sample_users\n',user);
        delete(fullfile('images',filenames{i}));
    end
end
