function step3bb_plot_validation_fmri_meg_combo(bids_dir, varargin)
    %% Function that plots the results of the combination analysis between fMRI and MEG
    % we are taking the univariate activation in V1 and FFA and use the
    % time-resolved MEG data to predict these
    %
    % @ Lina Teichmann, 2022
    %
    % Usage:
    % step3bb_plot_validation_fmri_meg_combo(bids_dir, ...)
    %
    % Inputs:
    %   bids_dir        path to the bids root folder 
    %
    % Returns:
    %   _               Figure in BIDS/derivatives folder


    %% folders
    res_dir         = [bids_dir '/derivatives/output/']; 
    figdir          = [bids_dir '/derivatives/figures/'];
    
    n_participants  = 4;


    % plotting parameters
    col_pp          = [0.21528455710115266, 0.5919540462603717, 0.3825837270552851;
                         0.24756252096251694, 0.43757475330612905, 0.5968141290988245;
                         0.7153368599631209, 0.546895038817448, 0.1270092896093349;
                         0.6772691643574462, 0.3168004639904812, 0.3167958318320575];

    col_pp_light    = [0.6020264172614653, 0.8666010337189269, 0.7198621708097467;
                        0.6329411764705883, 0.7552941176470587, 0.8572549019607842;
                        0.9347450980392157, 0.8266666666666667, 0.5554509803921569;
                        0.9019607843137256, 0.6803921568627451, 0.6803921568627451];

    x_size          = 0.19;
    y_size          = 0.15;
    x_pos           = linspace(0.1,0.9-x_size,4);

    %% load  results
    ffa_res = [];v1_res = [];
    for p = 1:n_participants
        tmp=table2array(readtable([res_dir,'/validation_fMRI-MEG-regression_ffa_P',num2str(p),'.csv'],'ReadVariableNames',1,'PreserveVariableNames',1));
        ffa_res(:,:,p) = tmp(:,2:end);

        tmp=table2array(readtable([res_dir,'/validation_fMRI-MEG-regression_v1_P',num2str(p),'.csv'],'ReadVariableNames',1,'PreserveVariableNames',1));
        v1_res(:,:,p) = tmp(:,2:end);

    end

    % load one example output file to get the time vector
    load([res_dir '/pairwise_decoding/P1_pairwise_decoding_1854_block1.mat'], 'res')
    tv = res.a.fdim.values{1}*1000;


    %% plot
    f = figure(1);clf
    f.Position=[0,0,600,700];
    for p = 1:n_participants
        [bci_ffa,~] = bootci(10000,{@mean,ffa_res(:,:,p)'},'alpha',.05,'type','per');
        [bci_v1,~] = bootci(10000,{@mean,v1_res(:,:,p)'},'alpha',.05,'type','per');

        % plot data with shaded bootci confidence intervals
        ax1 = axes('Position',[x_pos(p),0.5,x_size,y_size],'Units','normalized');hold on

        fill([tv,fliplr(tv)],[bci_ffa(1,:),fliplr(bci_ffa(2,:))],col_pp(p,:),'FaceAlpha',0.4,'EdgeColor',col_pp(p,:),'LineStyle','none')
        a(1)=plot(tv,movmean(mean(ffa_res(:,:,p),2),5),'Color',col_pp(p,:),'LineWidth',2);

        fill([tv,fliplr(tv)],[bci_v1(1,:),fliplr(bci_v1(2,:))],col_pp_light(p,:),'FaceAlpha',0.4,'EdgeColor',col_pp_light(p,:),'LineStyle','none')
        a(2)=plot(tv,movmean(mean(v1_res(:,:,p),2),5),'Color',col_pp_light(p,:),'LineWidth',2);

        plot(tv,tv*0,'k--')

        % make it look pretty
        xlim([tv(1),tv(end)])
        ylim([-0.1,.25])

        legend(a,[{'FFA'},{'V1'}],'box','off')

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

        set(ax1,'FontSize',14,'box','off','FontName','Helvetica');

    end


    % Plot differences 
    clear('a')
    for p = 1:4
        toplot = v1_res(:,:,p)-ffa_res(:,:,p);
        [bci,~] = bootci(10000,{@mean,toplot'},'alpha',.05,'type','per');

        % plot data with shaded bootci confidence intervals
        ax1 = axes('Position',[x_pos(p),0.5-y_size*1.5,x_size,y_size],'Units','normalized');hold on

        fill([tv,fliplr(tv)],[bci(1,:),fliplr(bci(2,:))],'k','FaceAlpha',0.4,'EdgeColor',col_pp(p,:),'LineStyle','none')
        a(1)=plot(tv,movmean(mean(toplot,2),5),'Color','k','LineWidth',2);

        plot(tv,tv*0,'k--')

        % make it look pretty
        xlim([tv(1),tv(end)])
        ylim([-0.15,.25])

        % add labels
        if p ==1
            ax1.YLabel.String = 'V1 - FFA';
        else
            ax1.YTick = [];
        end
        ax1.XLabel.String = 'time (ms)';

        set(ax1,'FontSize',14,'box','off','FontName','Helvetica');

        [highest,idx] = max(movmean(mean(toplot,2),5));
        ah = annotation('textarrow','X',[tv(idx)+200,tv(idx)+25],'Y',[highest,highest],'String',[num2str(tv(idx)),' ms'],'HorizontalAlignment','left','FontName','Helvetica','FontSize',14);
        set(ah,'parent',ax1); 
    end


    %% save figure
    fn = [figdir,'/validation_fmri-meg-combo'];
    tn = tempname;
    print(gcf,'-dpng','-r500',tn)

    im=imread([tn '.png']);
    [i,j]=find(mean(im,3)<255);margin=0;
    imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');


    print([fn '.pdf'],'-dpdf')

