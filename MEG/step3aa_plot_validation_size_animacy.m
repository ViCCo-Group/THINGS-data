function step3aa_plot_validation_size_animacy(bids_dir, varargin)
    %% Function that plots the results of the animacy & size validation analysis 
    %
    % @ Lina Teichmann, 2022
    %
    % Usage:
    % step3aa_plot_validation_size_animacy(bids_dir, ...)
    %
    % Inputs:
    %   bids_dir        path to the bids root folder 
    %
    % Returns:
    %   _               Figure in BIDS/derivatives folder
    % 
 
    
    %% parameters    
    figdir          = [bids_dir '/derivatives/figures/'];
    res_dir         = [bids_dir '/derivatives/output/'];

    n_participants  = 4;

    % plotting parameters
    col_pp          = [0.21528455710115266, 0.5919540462603717, 0.3825837270552851;
                        0.24756252096251694, 0.43757475330612905, 0.5968141290988245;
                        0.7153368599631209, 0.546895038817448, 0.1270092896093349;
                        0.6772691643574462, 0.3168004639904812, 0.3167958318320575];
                    
    x_size          = 0.19;
    y_size          = 0.15;
    x_pos           = linspace(0.1,0.9-x_size,4);
    y_pos           = [0.55, 0.55-y_size*2];     
    fontsize        = 20;
        
    %% load results                     
    % load  results
    res_animacy = [];res_size = [];
    for p = 1:n_participants
        tmp=table2array(readtable([res_dir,'/validation-animacy-P',num2str(p),'.csv'],'ReadVariableNames',1,'PreserveVariableNames',1));
        tmp=tmp(:,2:end);
        res_animacy(:,:,p) = mean(tmp,2);
        
        tmp=table2array(readtable([res_dir,'/validation-size-P',num2str(p),'.csv'],'ReadVariableNames',1,'PreserveVariableNames',1));
        tmp=tmp(:,2:end);
        res_size(:,:,p) = mean(tmp,2);
    end

    % load one example output file to get the time vector
    load([res_dir '/pairwise_decoding/P1_pairwise_decoding_1854_block1.mat'], 'res')
    tv = res.a.fdim.values{1}*1000;


    %% plot
    f = figure(1);clf
    f.Position=[0,0,600,700];

    text(0.5,0.39,'Size','FontSize',fontsize,'FontName','Helvetica','Units','normalized','HorizontalAlignment','center');
    text(0.5,0.75,'Animacy','FontSize',fontsize,'FontName','Helvetica','Units','normalized','HorizontalAlignment','center');
    axis off

    toplot = [{res_animacy},{res_size}];
    for row = 1:2
        for p = 1:n_participants

            % define threshold based on pre-stimulus onset
            max_preonset = max(toplot{row}(tv<=0,p));

            % plot data for each participant, fill when r > threshold
            ax1 = axes('Position',[x_pos(p),y_pos(row),x_size,y_size],'Units','normalized');

            plot(tv,toplot{row}(:,p),'LineWidth',2,'Color',col_pp(p,:));hold on
            hf = fill([tv,tv(end)],[max(toplot{row}(:,p),max_preonset);max_preonset],col_pp(p,:),'EdgeColor','none','FaceAlpha',0.2);

            % make it look pretty
            ylim([-0.1,.3])
            xlim([tv(1),tv(end)])

            % find onset of the longest shaded cluster
            i=reshape(find(diff([0;toplot{row}(:,p)>max_preonset;0])~=0),2,[]);
            [~,jmax]=max(diff(i));
            onset_idx=i(1,jmax);

            onset = tv(onset_idx); 

            % add a marker for onsets
            text(onset,gca().YLim(1), char(8593),'Color',col_pp(p,:), 'FontSize', 24, 'VerticalAlignment', 'bottom', 'HorizontalAlignment','Center','FontName','DejaVu Sans')
            text(onset+15,gca().YLim(1), [num2str(onset) ' ms'],'Color',col_pp(p,:), 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment','left')
            set(ax1,'FontSize',14,'box','off','FontName','Helvetica');

            % add subject title
            ax1_title = axes('Position',[x_pos(p)+0.001,y_pos(row)+y_size-0.01,0.03,0.03]); 
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
    end

    % save figure
    fn = [figdir,'/validation_size-animacy'];
    tn = tempname;
    print(gcf,'-dpng','-r500',tn)

    im=imread([tn '.png']);
    [i,j]=find(mean(im,3)<255);margin=0;
    imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');

    print([fn '.pdf'],'-dpdf')
    
end

