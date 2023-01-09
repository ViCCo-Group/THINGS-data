function step3f_validation_pairwise_decoding_stack_plot(bids_dir, toolbox_dir, imagewise_nblocks, sessionwise_nblocks, varargin)
    %% Stacking the pairwise decoding results and making a plot
    % 
    %
    % @ Lina Teichmann, 2022
    %
    % Usage:
    % step3f_validation_pairwise_decoding_stack_plot(bids_dir, imagewise_nblocks, sessionwise_nblocks, ...)
    %
    % Inputs:
    %   bids_dir            path to the bids root folder 
    %   toolbox_dir         path to toolbox folder containtining CoSMoMVPA
    %   imagewise_nblocks   number of blocks used to run the decoding analysis for the 200 objects in parallel
    %   sessionwise_nblocks number of blocks used to run the decoding analysis for the 1854 objects in parallel
    %
    % Returns:
    %   RDM200              stacked results of the decoding analysis for 200 images saved in BIDS/derivatives
    %   RDM1854             stacked results of the decoding analysis for 1854 concepts saved in BIDS/derivatives
    %   _                   Figure in BIDS/derivatives folder


    %% folders
    res_dir         = [bids_dir '/derivatives/output/']; 
    figdir          = [bids_dir '/derivatives/figures/'];
    
    addpath(genpath([toolbox_dir '/CoSMoMVPA']))

    %% parameters    
    n_participants  = 4;
    col_pp          = [0.21528455710115266, 0.5919540462603717, 0.3825837270552851;
                         0.24756252096251694, 0.43757475330612905, 0.5968141290988245;
                         0.7153368599631209, 0.546895038817448, 0.1270092896093349;
                         0.6772691643574462, 0.3168004639904812, 0.3167958318320575];

    %% load and stack
    for p = 1:n_participants
        for blocknr = 1:imagewise_nblocks
            outfn = [res_dir '/pairwise_decoding/P' num2str(p) '_pairwise_decoding_200_block' num2str(blocknr) '.mat'];
            load(outfn, 'res_pairs')        
            all_res{blocknr} = res_pairs;
        end
        res_200(p)= cosmo_stack(all_res);  


        for blocknr = 1:sessionwise_nblocks
            outfn = [res_dir '/pairwise_decoding/P' num2str(p) '_pairwise_decoding_1854_block' num2str(blocknr) '.mat'];
            load(outfn, 'res')  
            all_res_1854{blocknr} = res;
        end
        res_1854(p) = cosmo_stack(all_res_1854);
    end

    %% stack all pairwise decoding results
    for p = 1:n_participants
        all_res_samples_200(:,:,p)=res_200(p).samples;
        all_res_samples_1854(:,:,p)=res_1854(p).samples;
    end
    all_res_samples_200 = mean(all_res_samples_200,3);
    mean_rdm_200 = res_200(1);
    mean_rdm_200.samples = all_res_samples_200;

    all_res_samples_1854 = mean(all_res_samples_1854,3);
    mean_rdm_1854 = res_1854(1);
    mean_rdm_1854.samples = all_res_samples_1854;


    %% plot mean accuracy for 200 & 1854 object pairwise decoding over time
    figure(1);clf

    tv = res_200(1).a.fdim.values{1}*1000;
    
    plot_mean_decoding(1,res_200,'Mean Pairwise Decoding 200 Objects',tv,n_participants,col_pp,[45,80])
    %%
    saveas(gcf,[figdir '/PairwiseDecoding_timeseries_200.pdf'])

    figure(2);clf
    tv = res_1854(1).a.fdim.values{1}*1000;
    plot_mean_decoding(1,res_1854,'Mean Pairwise Decoding 1854 Objects',tv,n_participants,col_pp,[48,60])
    saveas(gcf,[figdir '/PairwiseDecoding_timeseries_1854.pdf'])

    %% plot the dissimilarity matrix 200
    figure(3);clf
    mat = nan(200,200,length(tv),n_participants);
    all_combinations = combnk(1:200,2);
    for p = 1:n_participants
        for t = 1:length(tv)
            for i = 1:size(all_combinations,1)
                r = all_combinations(i,1);
                c = all_combinations(i,2);

                mat(r,c,t,p) = res_200(p).samples(i,t);
                mat(c,r,t,p) = res_200(p).samples(i,t);

            end

        end
    end

    average_rdm = mean(mat,4); 
    plot_rdm(2,res_200,average_rdm,'Pairwise Decoding (peak time), 200 Objects',n_participants,[45,90])
    saveas(gcf,[figdir '/PairwiseDecoding_rdm_200.pdf'])

    save([res_dir,'/validation-pairwise_decoding_RDM200'],'mat','-v7.3')


    %% plot the dissimilarity matrix 1854
    figure(4);clf
    mat = nan(1854,1854,length(tv),n_participants);
    for p = 1:n_participants
        disp(p)
        all_combinations = [res_1854(p).sa.target1,res_1854(p).sa.target2];

        for t = 1:length(tv)
            for i = 1:size(all_combinations,1)
                r = all_combinations(i,1);
                c = all_combinations(i,2);
                mat(r,c,t,p) = res_1854(p).samples(i,t);
                mat(c,r,t,p) = res_1854(p).samples(i,t);
            end

        end
    end

    average_rdm = mean(mat,4); 
    plot_rdm(2,res_1854,average_rdm,'Pairwise Decoding (peak time), 1854 Objects',n_participants,[45,70])
    saveas(gcf,[figdir '/PairwiseDecoding_rdm_1854.pdf'])

    save([res_dir,'/validation-pairwise_decoding_RDM1854'],'mat','-v7.3')


end


%% Helper functions for plotting
function plot_mean_decoding(fignum,toplot,title_string,tv,n_participants,cols,ylimit)
    figure(fignum);clf;

    for p = 1:n_participants
        a(p) = plot(tv,mean(toplot(p).samples*100),'LineWidth',2,'Color',cols(p,:));hold on
    end

    plot(tv,tv*0+50,'k--')

    xlabel('time (ms)')
    ylabel('Decoding Accuracy (%)')
    title(title_string)
    set(gcf,'Units','centimeters','Position',[0,0,15,10])
    set(gca,'FontSize',14,'Box','off','FontName','Helvetica')
    legend(a,['M1';'M2';'M3';'M4'])

    xlim([tv(1),tv(end)])
    ylim([ylimit(1),ylimit(2)])

end


function plot_rdm(fignum,toplot,average_rdm,title_string,n_participants,col_lim)
    figure(fignum);clf;

    for p = 1:n_participants
        all_res(p,:)=mean(toplot(p).samples);
    end
    [~,max_index] = max(mean(all_res));
    imagesc(average_rdm(:,:,max_index)*100,col_lim); cb=colorbar; cb.Label.String = 'Decoding Accuracy (%)';cb.Location = 'southoutside';
    axis square;
    set(gca,'xTick',[],'yTick',[],'FontSize',12,'FontName','Helvetica')
    title(title_string)
    set(gcf,'Units','centimeters','Position',[0,0,12,12])

end

    
    
