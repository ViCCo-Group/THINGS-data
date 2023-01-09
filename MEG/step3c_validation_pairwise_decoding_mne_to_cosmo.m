function step3c_validation_pairwise_decoding_mne_to_cosmo(bids_dir, toolbox_dir, varargin)
    %% Function that takes preprocessed MNE data and transforms it to a matlab-cosmo script
    % Note: this script requires the MNE-matlab & cosmomvpa toolboxes 
    %
    % @ Lina Teichmann, 2022
    %
    % Usage:
    % step3c_validation_pairwise_decoding_mne_to_cosmo(bids_dir, ...)
    %
    % Inputs:
    %   bids_dir        path to the bids root folder 
    %   toolbox_dir         path to toolbox folder containtining CoSMoMVPA
    %
    % Returns:
    %   ds              Cosmo data struct, saved in BIDS/derivatives/preprocessed folder
    % 
    

    %% parameters    
    preprocdir      = [bids_dir '/derivatives/preprocessed/'];

    addpath(genpath([toolbox_dir '/mne-matlab']))
    addpath(genpath([toolbox_dir '/CoSMoMVPA']))
    
    n_participants  = 4;

    %% loop
    for p=1:n_participants
        tic
        tmp_filenames = dir([preprocdir '/preprocessed_P' num2str(p) '-epo*.fif']);  
        n1 = {tmp_filenames.name};
        [~,I] = sort(cellfun(@length,n1));
        all_files = n1(I);
        
        %sanity check
        disp('stacking files in this order: ')
        for i = 1:length(all_files); disp(all_files{i}); end
        
        for f = 1:length(all_files)
            epo{f} = make_ds(fiff_read_epochs([preprocdir filesep cell2mat(all_files(f))]));
        end

        sa_tab = readtable([bids_dir '/sourcedata/sample_attributes_P' num2str(p) '.csv']);
        sa = table2struct(sa_tab,'toscalar',1);

        ds = cosmo_stack(epo);
        ds.sa = sa;
        
        save([preprocdir '/P' num2str(p) '_cosmofile.mat'],'ds','-v7.3')
        fprintf('Saving finished in %i seconds\n',ceil(toc))

    end



end


%% helper function
function ds = make_ds(part)
    data = reshape(part.data,[size(part.data,1),size(part.data,2)*size(part.data,3)]);
    chan = repmat(1:size(part.data,2),1,size(part.data,3));
    time = repelem(1:size(part.data,3),1,size(part.data,2));

    ds = struct();
    ds.samples = data;
    ds.a.fdim.labels = [{'chan'};{'time'}];
    ds.a.fdim.values = [{1:size(part.data,2)};{part.times}];
    ds.fa.chan = chan; 
    ds.fa.time = time;   
end

    
