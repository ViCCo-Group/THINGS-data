function step3g_validation_pairwise_decoding_mds(bids_dir, toolbox_dir, varargin)
    %% Making an MDS plot for some image categories
    % 
    %
    % @ Lina Teichmann, 2022
    %
    % Usage:
    % step3g_validation_pairwise_decoding_mds(bids_dir, ...)
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
    colour_gray     = [0.75,0.75,0.75];
    line_col        = [0, 109, 163]./255;
    contrast_cols	= cat(3,[[70, 225, 240]./255;[191, 82, 124]./255],...
                        [[126, 92, 150]./255;[255, 153, 0]./255]);        
    ylims           = [41,80;45,65];
    x_size          = 0.19;
    y_size          = 0.15;
    size_mds        = 0.09;
    x_pos           = linspace(0.1,0.9-x_size,4);
    y_pos           = [0.8,0.54,0.29];


    %% load results
    % load one example output file to get the time vector
    load([res_dir '/pairwise_decoding/P1_pairwise_decoding_1854_block1.mat'], 'res')
    tv = res.a.fdim.values{1}*1000;

    % load decoding results
    load([res_dir,'/validation-pairwise_decoding_RDM1854'],'mat')
    decoding_1854 = mat; 

    load([res_dir,'/validation-pairwise_decoding_RDM200'],'mat')
    decoding_200 = mat; 

    % load category labels
    labels = readtable([bids_dir '/sourcedata/category_mat_manual.tsv'],'FileType','text','PreserveVariableNames',1);


    %% MDS: highlighting animals vs food and vehicles vs tools
    colour_cat1 = zeros(1854,1);
    colour_cat1(labels.animal==1&labels.food==0)=1;
    colour_cat1(labels.animal==0&labels.food==1)=2;

    colour_cat2 = zeros(1854,1);
    colour_cat2(labels.vehicle==1&labels.tool==0)=1;
    colour_cat2(labels.vehicle==0&labels.tool==1)=2;
%     colour_cat2(labels.plant==1&labels.bodyPart==0)=1;
%     colour_cat2(labels.plant==0&labels.bodyPart==1)=2;


    mds_categories = [colour_cat1,colour_cat2];
    legends = cat(3,[{'animals'},{'food'}],[{'vehicles'},{'tools'}]);
%     legends = cat(3,[{'animals'},{'food'}],[{'plants'},{'body parts'}]);

    average_rdm = mean(decoding_1854,4);

    % loop over the two MDS comparisons
    for i = 1:size(mds_categories,2)
        colour_cat = mds_categories(:,i);
        mean_mds_comp = squeeze(mean(mean(average_rdm(colour_cat==1,colour_cat==2,:))));

        avg = []; 
        for t=1:length(tv)
            D = average_rdm(:,:,t);
            D(find(eye(size(D))))=0;
            [Y(:,:,t),~] = cmdscale(D,2);

            tmp = triu(average_rdm(:,:,t),1);
            tmp = tmp(colour_cat==0,colour_cat==0);
            avg=[avg;mean(mean(tmp(tmp>0)))];

        end

        % use procrustes to align the different MDS over time
        for t = length(tv):-1:2
            [~,z(:,:,t,i)] = procrustes(Y(:,:,t),Y(:,:,t-1));
        end

    end

    %% plot timecourse together with MDS snapshots
    toplot = zeros(length(tv),4);

    all_decoding = [{decoding_200},{decoding_1854}];

    for i = 1:2
        for p = 1:n_participants
            for t = 1:length(tv)
                tmp = triu(all_decoding{i}(:,:,t,p),1);
                toplot(t,p,i) = mean(mean(tmp(tmp>0)))*100;
            end
        end
    end

    f=figure(2);clf; 
    f.Position=[0,0,600,700];
    titles = [{'Object image decoding'},{'Object category decoding'}];



    % decoding plots for image and category decoding
    for i = 1:2
        for p = 1:4

            % define threshold based on pre-stimulus onset
            max_preonset = max(toplot(tv<=0,p,i));

            % plot data for each participant, fill when r > threshold
            disp(y_pos(i))
            ax = axes('Position',[x_pos(p),y_pos(i),x_size,y_size],'Units','normalized');
            plot(tv,toplot(:,p,i),'LineWidth',2,'Color',col_pp(p,:));hold on
            hf = fill([tv,tv(end)],[max(toplot(:,p,i),max_preonset);max_preonset],col_pp(p,:),'EdgeColor','none','FaceAlpha',0.2);

            % make it look pretty
            ylim(ylims(i,:))
            xlim([tv(1),tv(end)])
            if p ==1
                ax.YLabel.String = [{'Decoding'}; {'accuracy (%)'}];
            else
                ax.YTick = [];
            end
            xlabel('time (ms)')
            set(gca(),'FontSize',12,'box','off','FontName','Helvetica');
            
            % find onset of the longest shaded cluster
            ii=reshape(find(diff([0;toplot(:,p,i)>max_preonset;0])~=0),2,[]);
            [~,jmax]=max(diff(ii));
            onset_idx=ii(1,jmax);

            onset = tv(onset_idx);

            % add a marker for onsets
            text(onset,gca().YLim(1), char(8593),'Color',col_pp(p,:), 'FontSize', 24, 'VerticalAlignment', 'bottom', 'HorizontalAlignment','Center','FontName','Helvetica')
            text(onset+20,gca().YLim(1), [num2str(onset) ' ms'],'Color',col_pp(p,:), 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment','left')

            % add subject title
            ax1_title = axes('Position',[x_pos(p)+0.001,y_pos(i)+y_size-0.01,0.03,0.03]); 
            text(0,0,['M' num2str(p)],'FontSize',11,'FontName','Helvetica');
            ax1_title.Visible = 'off'; 

        end

    end


    % add title
    row_title = axes('Position',[x_pos(1)+0.01,y_pos(1)+y_size+0.02,0.03,0.03]); 
    text(0,0,titles{1},'FontSize',14,'FontWeight','bold','FontName','Helvetica')
    row_title.Visible = 'off'; 

    row_title = axes('Position',[x_pos(1)+0.01,y_pos(2)+y_size+0.04,0.03,0.03]); 
    text(0,0,titles{2},'FontSize',14,'FontWeight','bold','FontName','Helvetica')
    row_title.Visible = 'off'; 

    row_title = axes('Position',[x_pos(1)+0.01,y_pos(2)+y_size+0.015,0.03,0.03]); 
    text(0,0,'Single subjects','FontSize',12,'FontName','Helvetica')
    row_title.Visible = 'off'; 


    % MDS
    group_avg = mean(toplot(:,:,2),2);
    ax1 = axes('Position',[x_pos(1),y_pos(3),x_pos(end)+x_size/2,y_size],'Units','normalized');
    upper = group_avg' + std(toplot(:,:,2)')/sqrt(size(toplot,2));
    lower = group_avg' - std(toplot(:,:,2)')/sqrt(size(toplot,2));

    fill([tv,fliplr(tv)],[lower,fliplr(upper)],'k','FaceAlpha',0.1,'LineStyle','none'); hold on
    plot(tv, group_avg,'Color','k','LineWidth',2);
    plot(tv,tv*0+50,'k--')

    fill([tv,fliplr(tv)],[lower,fliplr(upper)],[1,1,1]/255,'FaceAlpha',0.1,'LineStyle','none'); hold on
    plot(tv, group_avg,'Color',[1,1,1]/255,'LineWidth',2);
    plot(tv,tv*0+50,'Color',[1,1,1]/255)

    xlim([tv(1),tv(end)])

    ax1.YLabel.String = [{'Decoding'}; {'accuracy (%)'}];
    ax1.XLabel.String='time (ms)';
    ax1.XTick = [-80,0,120,320,520,720,920,1120];
    set(ax1,'FontSize',12,'box','off','FontName','Helvetica');

    % add title
    row_title = axes('Position',[x_pos(1)+0.01,y_pos(3)+y_size+0.02,0.03,0.03]); 
    text(0,0,'Group Average','FontSize',12,'FontName','Helvetica')
    row_title.Visible = 'off'; 


    t_idx = 5:40:length(tv)+1;
    t_time = tv(t_idx); 
    tv_pix = linspace(ax1.Position(1),ax1.Position(1)+ax1.Position(3),length(tv));

    % loop over MDS comparisons
    for i = 1:2
        color1 = contrast_cols(1,:,i);
        color2 = contrast_cols(2,:,i);
        colour_cat = mds_categories(:,i);

        % loop over time
        for t = 1:length(t_idx)
            ax2 = axes();
            a=[];
            a(3)=scatter(z(colour_cat==0,1,t_idx(t),i),z(colour_cat==0,2,t_idx(t),i),15,'MarkerFaceAlpha',1,'MarkerFaceColor',colour_gray,'MarkerEdgeColor','None');hold on
            a(2)=scatter(z(colour_cat==1,1,t_idx(t),i),z(colour_cat==1,2,t_idx(t),i),15,'MarkerFaceAlpha',0.7,'MarkerFaceColor',color1,'MarkerEdgeColor','None');hold on
            a(1)=scatter(z(colour_cat==2,1,t_idx(t),i),z(colour_cat==2,2,t_idx(t),i),15,'MarkerFaceAlpha',0.5,'MarkerFaceColor',color2,'MarkerEdgeColor','None');hold on

            ax2.XTick = [];
            ax2.YTick = [];
            ax2.XLim=[-0.3,0.3];
            ax2.YLim=[-0.3,0.3];
            axis square

            if i == 1
                ax2.Position = [tv_pix(t_idx(t))-size_mds/2,y_pos(3)-0.16,size_mds,size_mds];
            else
                ax2.Position = [tv_pix(t_idx(t))-size_mds/2,y_pos(3)-0.16-size_mds-0.01,size_mds,size_mds];
            end
            annotation('arrow',[tv_pix(t_idx(t)),tv_pix(t_idx(t))],[y_pos(3)-0.025,y_pos(3)-0.16+size_mds])
            set(ax2,'XColor', 'none','YColor','none')
        end

        ax2 = axes('box','off');hold on
        if i == 1
            ax2.Position = [0.92-0.05,y_pos(3)-0.16,size_mds,size_mds];
        else
            ax2.Position = [0.92-0.05,y_pos(3)-0.16-size_mds-0.01,size_mds,size_mds];
        end
        plot(0.1,0.7,'k.','MarkerSize',25,'Color',color1)
        text(0.2,0.7,legends(:,1,i),'FontSize',12,'FontName','Helvetica','HorizontalAlignment','left','VerticalAlignment','middle')
        plot(0.1,0.5,'k.','MarkerSize',25,'Color',color2)
        text(0.2,0.5,legends(:,2,i),'FontSize',12,'FontName','Helvetica','HorizontalAlignment','left','VerticalAlignment','middle')
        plot(0.1,0.3,'k.','MarkerSize',25,'Color',colour_gray)
        text(0.2,0.3,'all other','FontSize',12,'FontName','Helvetica','HorizontalAlignment','left','VerticalAlignment','middle')
        ax2.Visible = 'off';
        ax2.XLim = [0,1];
        ax2.YLim = [0,1];
    end    

    % save figure
    fn = [figdir,'/validation_decoding-mds'];
    tn = tempname;
    print(gcf,'-dpng','-r500',tn)

    im=imread([tn '.png']);
    [i,j]=find(mean(im,3)<255);margin=0;
    imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');
    print([fn '.pdf'],'-dpdf')


end
