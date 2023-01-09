function step3d_validation_pairwise_decoding200(bids_dir, toolbox_dir, participant, blocknr, n_blocks)

    %% Function to run pairwise decoding analysis for the 200 repeat stimuli
    %
    % @ Lina Teichmann, 2022
    %
    % Usage:
    % step3d_validation_pairwise_decoding200(bids_dir,participant, ...)
    %
    % Inputs:
    %   bids_dir        path to the bids root folder 
    %   toolbox_dir         path to toolbox folder containtining CoSMoMVPA
    %   participant     participant number
    %   blocknr         number of the chunk you want to run this analysis for 
    %   n_blocks        how many blocks you want to run this analysis in (this is to make it faster by running stuff in parallel)
    %
    % Returns:
    %   decoding_acc    file that has the decoding accuracy for each decoding block ('PX_pairwise)decoding_200_blockX.mat')
    %   decoding_pairs  file that contains which pairwise comparisons were run so it can be stacked back together ('PX_pairwise)decoding_200_blockX_pairs.mat')
    
    
    %% folders
    preprocdir      = [bids_dir '/derivatives/preprocessed/'];
    res_dir         = [bids_dir '/derivatives/output/']; 
    
    addpath(genpath([toolbox_dir '/CoSMoMVPA']))

    load([preprocdir '/P' num2str(participant) '_cosmofile.mat'],'ds');
    
    % make a pairwise decoding folder if it does not exist
    if ~exist([res_dir '/pairwise_decoding'], 'dir')
        mkdir([res_dir '/pairwise_decoding'])
    end
    outfn = [res_dir '/pairwise_decoding/P' num2str(participant) '_pairwise_decoding_200_block' num2str(blocknr) '.mat'];
    outfn_pairs = [res_dir '/pairwise_decoding/P' num2str(participant) '_pairwise_decoding_200_block' num2str(blocknr) '_pairs.mat'];
    
    
    %% pairwise decoding
    ds = cosmo_slice(ds,strcmp(ds.sa.trial_type,'test'));
    ds.sa.targets = ds.sa.things_category_nr;
    ds.sa.chunks = ds.sa.session_nr;
    all_combinations = combnk(unique(ds.sa.targets),2);

    % split into blocks
    step = ceil(length(all_combinations)/n_blocks);
    s = 1:step:length(all_combinations);
    blocks = cell(length(s),1);
    for b = 1:length(s)
        blocks{b} = all_combinations(s(b):min(s(b)+step-1,length(all_combinations)),:);
    end

    combs = blocks{blocknr};
    save(outfn_pairs, 'combs')
    nproc = cosmo_parallel_get_nproc_available;

    res = [];
    for pairs = 1:length(combs)
        tic
        disp([num2str(pairs) ' out of ' num2str(length(combs))])
        ds_p = cosmo_slice(ds,ismember(ds.sa.things_category_nr,combs(pairs,:)));
        partitions = cosmo_nfold_partitioner(ds_p); 
        measure_args=struct(); 
        measure_args.classifier=@cosmo_classify_lda;
        measure_args.partitions=partitions;
        measure_args.nproc = nproc;
        nbrhood=cosmo_interval_neighborhood(ds_p,'time','radius',0);
        res{pairs}=cosmo_searchlight(ds_p,nbrhood,@cosmo_crossvalidation_measure,measure_args);
        toc
    end
    res_pairs = cosmo_stack(res);
    save(outfn, 'res_pairs','-v7.3')
 
end


