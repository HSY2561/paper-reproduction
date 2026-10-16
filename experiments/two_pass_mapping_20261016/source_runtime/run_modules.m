% Reproduction harness only. Calls unchanged author functions; not a new planner.
function run_modules
root = fileparts(mfilename('fullpath'));
cd(fullfile(root, 'author_software'));
set(groot, 'defaultFigureVisible', 'off');
global spdist s allNode a total_length
spdist = @(P,Ps) sqrt((P(1,1)-Ps(:,1)).^2 + (P(1,2)-Ps(:,2)).^2);
total_length = @(Ps) sum(sqrt(sum(diff(Ps).*diff(Ps),2)));
s = fix(4.5*12/2*25.4/2); a = fix(7/2*25.4/2);
allNode = [0 0;2896 3048];
areaPoly = polyshape([0 3048 3048 0], [0 0 2896 2896]);
[route, flag] = BoustrophedonPath(areaPoly, areaPoly, [1 2], s/sqrt(2), [], [0 0], false, false, false);
assert(~isempty(route) && all(isfinite(route), 'all'));
writematrix(route, fullfile(root,'boustrophedon_path_px.csv'));
save(fullfile(root,'boustrophedon_result.mat'),'route','flag','s','a');
fig=figure('Position',[50 50 1000 850]);
plot(areaPoly,'FaceColor',[.95 .95 .95],'FaceAlpha',1); hold on;
plot(route(:,1),route(:,2),'-o','Color',[0 .4 .75],'LineWidth',1.8,'MarkerSize',3);
scatter(route(1,1),route(1,2),90,[.1 .65 .25],'filled');
scatter(route(end,1),route(end,2),90,[.85 .2 .15],'filled');
axis equal; axis tight; grid on; xlabel('x (author map pixels)'); ylabel('y (author map pixels)');
title('Author BoustrophedonPath: rectangle coverage module');
subtitle('Synthetic empty workspace; no camera calibration and no robot execution');
exportgraphics(fig,fullfile(root,'boustrophedon_official_module.png'),'Resolution',180);
meta=struct('status','success','waypoints',size(route,1),'path_length_px',total_length(route),'sensor_radius_px',s,'scan_step_px',2*s/sqrt(2),'units','author map pixels','scope','Unmodified author BoustrophedonPath on synthetic rectangle; not measured robot trajectory');
fid=fopen(fullfile(root,'boustrophedon_metrics.json'),'w'); fprintf(fid,'%s',jsonencode(meta,PrettyPrint=true)); fclose(fid);
disp(jsonencode(meta));
disp('BOUSTROPHEDON_COMPLETE');

% Real prediction from the existing P0 artifact. No ground truth or image edits.
inputFile='D:/论文复现/P0_real_segformer_trajectory/20160222_080850_1281_361_segformer_mask_raw.png';
mask=imread(inputFile); mask=mask(:,:,1)>0;
try
    [node,edgeList,ttt]=Image_planning_func_SCC(mask);
    save(fullfile(root,'segformer_author_graph.mat'),'node','edgeList','ttt');
    writematrix(node,fullfile(root,'segformer_graph_nodes_row_col.csv'));
    writematrix(edgeList,fullfile(root,'segformer_graph_edges.csv'));
    rawSkel=bwskel(bwmorph(mask,'fill'));
    prunedSkel=bwmorph(rawSkel,'spur',a);
    imwrite(rawSkel,fullfile(root,'segformer_raw_skeleton.png'));
    imwrite(prunedSkel,fullfile(root,'segformer_author_pruned_skeleton.png'));
    fig=figure('Position',[50 50 1500 480]);
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile; imshow(mask); title('Real SegFormer predicted mask');
    nexttile; imshow(prunedSkel); title(sprintf('Author skeleton + spur(%d)',a));
    nexttile; imshow(mask); hold on;
    for q=1:size(edgeList,1)
        ids=edgeList(q,1:2); plot(node(ids,2),node(ids,1),'-','Color',[1 .35 .1],'LineWidth',1.5);
    end
    if ~isempty(node), scatter(node(:,2),node(:,1),16,[0 .9 1],'filled'); end
    title('Author graph output (image pixels)');
    sgtitle('Interface smoke only: fixed author scale; no world map or calibrated repair path');
    exportgraphics(fig,fullfile(root,'segformer_author_graph_smoke.png'),'Resolution',180);
    out=struct('status','success','input',inputFile,'mask_shape',size(mask),'foreground_pixels',nnz(mask),'raw_skeleton_pixels',nnz(rawSkel),'author_pruned_skeleton_pixels',nnz(prunedSkel),'nodes',size(node,1),'edges',size(edgeList,1),'author_reported_compute_time',ttt,'units','image pixels, nodes are row/column','limitation','Author hardcodes 2 mm/px and 7-inch footprint; original spur threshold 44 px is NOT calibrated to this image. Graph is a footprint-planning abstraction, not exact crack centerline.');
    fid=fopen(fullfile(root,'segformer_author_graph_metrics.json'),'w'); fprintf(fid,'%s',jsonencode(out,PrettyPrint=true)); fclose(fid);
    disp(jsonencode(out));
    disp('SEGFORMER_INTERFACE_COMPLETE');
catch ME
    out=struct('status','failed','identifier',ME.identifier,'message',ME.message,'stack',ME.stack);
    fid=fopen(fullfile(root,'segformer_author_graph_metrics.json'),'w'); fprintf(fid,'%s',jsonencode(out,PrettyPrint=true)); fclose(fid);
    fprintf(2,'%s\n',getReport(ME,'extended','hyperlinks','off'));
    rethrow(ME);
end
end
