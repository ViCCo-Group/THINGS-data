function step3e_validation_pairwise_decoding1854(bids_dir, toolbox_dir, participant, blocknr, n_blocks)
  %% Function to run pairwise decoding analysis for the 1854 object classes
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
    %   decoding_acc    file that has the decoding accuracy for each decoding block ('PX_pairwise)decoding_1854_blockX.mat')
    %   decoding_pairs  file that contains which pairwise comparisons were run so it can be stacked back together ('PX_pairwise)decoding_1854_blockX_pairs.mat')
    
    
    
    %% folders
    preprocdir      = [bids_dir '/derivatives/preprocessed/'];
    res_dir         = [bids_dir '/derivatives/output/']; 
    
    addpath(genpath([toolbox_dir '/CoSMoMVPA']))

    load([preprocdir '/P' num2str(participant) '_cosmofile.mat'],'ds');
    
    % make a pairwise decoding folder if it does not exist
    if ~exist([res_dir '/pairwise_decoding'], 'dir')
        mkdir([res_dir '/pairwise_decoding'])
    end
    outfn = [res_dir '/pairwise_decoding/P' num2str(participant) '_pairwise_decoding_1854_block' num2str(blocknr) '.mat'];
    outfn_pairs = [res_dir '/pairwise_decoding/P' num2str(participant) '_pairwise_decoding_1854_block' num2str(blocknr) '_pairs.mat'];

    %% pairwise decoding
    ds = cosmo_slice(ds,strcmp(ds.sa.trial_type,'exp'));
    ds.sa.targets = ds.sa.things_category_nr;
    ds.sa.chunks = ds.sa.session_nr;
    all_combinations = combnk(unique(ds.sa.targets),2);
    all_targets = unique(ds.sa.targets);
    all_chunks = unique(ds.sa.chunks); 

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
  
    
     %% create RDM
    % find the items belonging to the exemplars
    target_idx = cell(1,length(all_targets));
    for j=1:length(all_targets)
        target_idx{j} = find(ds.sa.targets==all_targets(j));
    end
    % for each chunk, find items belonging to the test set
    test_chunk_idx = cell(1,length(all_chunks));
    for j=1:length(all_chunks)
        test_chunk_idx{j} = find(ds.sa.chunks==all_chunks(j));
    end
    
    %% make blocks for parfor loop
    step = ceil(length(combs)/nproc);
    s = 1:step:length(combs);
    comb_blocks = cell(1,length(s));
    for b = 1:nproc
        comb_blocks{b} = combs(s(b):min(s(b)+step-1,length(combs)),:);
    end
    
    %arguments for searchlight and crossvalidation
    ma = struct();
    ma.classifier = @cosmo_classify_lda;
    ma.output = 'accuracy';
    ma.check_partitions = false;
    ma.nproc = 1;
    ma.progress = 0;
    ma.partitions = struct();

    % set options for each worker process
    nh = cosmo_interval_neighborhood(ds,'time','radius',0);
    worker_opt_cell = cell(1,nproc);
    for procs=1:nproc
        worker_opt=struct();
        worker_opt.ds=ds;
        worker_opt.ma = ma;
        worker_opt.uc = all_chunks;
        worker_opt.worker_id=procs;
        worker_opt.nproc=nproc;
        worker_opt.nh=nh;
        worker_opt.combs = comb_blocks{procs};
        worker_opt.target_idx = target_idx;
        worker_opt.test_chunk_idx = test_chunk_idx;
        worker_opt_cell{procs}=worker_opt;
    end
    %% run the workers
    tic
    result_map_cell=cosmo_parcellfun(nproc,@run_block_with_worker,worker_opt_cell,'UniformOutput',false);
    toc
    %% cat the results
    res=cosmo_stack(result_map_cell);

    %% save
    fprintf('Saving...');tic
    save(outfn,'res','-v7.3')
    fprintf('Saving finished in %i seconds\n',ceil(toc))
  

end

function res_block = run_block_with_worker(worker_opt)
    ds=worker_opt.ds;
    nh=worker_opt.nh;
    ma=worker_opt.ma;
    uc=worker_opt.uc;
    target_idx=worker_opt.target_idx;
    test_chunk_idx=worker_opt.test_chunk_idx;
    worker_id=worker_opt.worker_id;
    nproc=worker_opt.nproc;
    combs=worker_opt.combs;
    res_cell = cell(1,length(combs));
    cc=clock();mm='';
    for i=1:length(combs)
        idx_ex = [target_idx{combs(i,1)}; target_idx{combs(i,2)}];
        [ma.partitions.train_indices,ma.partitions.test_indices] = deal(cell(1,length(uc)));
        for j=1:length(uc)
            ma.partitions.train_indices{j} = setdiff(idx_ex,test_chunk_idx{j});
            ma.partitions.test_indices{j} = intersect(test_chunk_idx{j},idx_ex);
        end
        res_cell{i} = cosmo_searchlight(ds,nh,@cosmo_crossvalidation_measure,ma);
        res_cell{i}.sa.target1 = combs(i,1);
        res_cell{i}.sa.target2 = combs(i,2);
        if ~mod(i,10)
            mm=cosmo_show_progress(cc,i/length(combs),sprintf('%i/%i for worker %i/%i\n',i,length(combs),worker_id,nproc),mm);
        end
    end
    res_block = cosmo_stack(res_cell);
end



