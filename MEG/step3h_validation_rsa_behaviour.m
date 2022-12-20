function step3h_validation_rsa_behaviour(bids_dir, toolbox_dir, varargin)
    %% RSA between the behavioural similarity matrix and the MEG pairwise decoding accuracies
    % 
    %
    % @ Lina Teichmann, 2022
    %
    % Usage:
    % step3h_validation_rsa_behaviour(bids_dir, ...)
    %
    % Inputs:
    %   bids_dir            path to the bids root folder 
    %   toolbox_dir         path to toolbox folder containtining CoSMoMVPA

    % Returns:
    %   _                   Figure in BIDS/derivatives folder


    %% folders
    res_dir         = [bids_dir '/derivatives/output/']; 
    figdir          = [bids_dir '/derivatives/figures/'];
    
    addpath(genpath([toolbox_dir '/CoSMoMVPA']))
    %% parameters    
    n_participants  = 4;

    % plotting parameters
    col_pp          = [0.21528455710115266, 0.5919540462603717, 0.3825837270552851;
                         0.24756252096251694, 0.43757475330612905, 0.5968141290988245;
                         0.7153368599631209, 0.546895038817448, 0.1270092896093349;
                         0.6772691643574462, 0.3168004639904812, 0.3167958318320575];

    x_size          = 0.19;
    y_size          = 0.15;
    x_pos           = linspace(0.1,0.9-x_size,4);

    %% load stuff                 
    % load behavioural similarities
    load([bids_dir '/sourcedata/spose_similarity.mat'],'spose_sim')

    % load decoding results
    load([res_dir,'/validation-pairwise_decoding_RDM1854'],'mat')
    decoding_1854 = mat; 

    load([res_dir,'/validation-pairwise_decoding_RDM200'],'mat')
    decoding_200 = mat; 

    % load one example output file to get the time vector
    load([res_dir '/pairwise_decoding/P1_pairwise_decoding_1854_block1.mat'], 'res')
    tv = res.a.fdim.values{1}*1000;


    %% RSA: behaviour - MEG
    corr_beh = zeros(size(decoding_1854,3),4);

    for p = 1:4
        for t = 1:size(decoding_1854,3)
            dat = decoding_1854(:,:,t,p);
            corr_beh(t,p)=corr(dat(:),spose_sim(:), 'rows','complete','Type','Pearson');
        end
    end

    save([res_dir '/validation_rsa-behaviour'],'corr_beh')

    %% plot 
    f = figure(1);clf
    f.Position=[0,0,600,700];

    for p = 1:n_participants
        % define threshold based on pre-stimulus onset
        max_preonset = max(corr_beh(tv<=0,p)*-1);

        % plot data for each participant, fill when r > threshold
        ax1 = axes('Position',[x_pos(p),0.5,x_size,y_size],'Units','normalized');
        plot(tv,corr_beh(:,p)*-1,'LineWidth',2,'Color',col_pp(p,:));hold on
        hf = fill([tv,tv(end)],[max(corr_beh(:,p)*-1,max_preonset);max_preonset],col_pp(p,:),'EdgeColor','none','FaceAlpha',0.2);

        % make it look pretty
        ylim([-0.03,.11])
        xlim([tv(1),tv(end)])

        % find onset of the longest shaded cluster
        i=reshape(find(diff([0;corr_beh(:,p)*-1>max_preonset;0])~=0),2,[]);
        [~,jmax]=max(diff(i));
        onset_idx=i(1,jmax);
        onset = tv(onset_idx); 

        % add a marker for onsets
        text(onset,gca().YLim(1), char(8593),'Color',col_pp(p,:), 'FontSize', 20, 'VerticalAlignment', 'bottom', 'HorizontalAlignment','Center','FontName','Helvetica')
        text(onset+15,gca().YLim(1), [num2str(onset) ' ms'],'Color',col_pp(p,:), 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment','left')
        set(ax1,'FontSize',14,'box','off','FontName','Helvetica');

        % add subject title
        ax1_title = axes('Position',[x_pos(p)+0.001,0.5+y_size-0.01,0.03,0.03]); 
        text(0,0,['M' num2str(p)],'FontSize',12,'FontName','Helvetica');
        ax1_title.Visible = 'off'; 

        % add labels    
        if p ==1
            ax1.YLabel.String = 'r';
        else
            ax1.YTick = [];
        end
        ax1.XLabel.String = 'time (ms)';

    end

    % save figure
    fn = [figdir,'/validation_rsa-behaviour'];
    tn = tempname;
    print(gcf,'-dpng','-r500',tn)

    im=imread([tn '.png']);
    [i,j]=find(mean(im,3)<255);margin=0;
    imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');

    print([fn '.pdf'],'-dpdf')

