clc
clearvars

% inputs
opt = globals();
root_dir = '/home/yihao/Research/SelfObjectPose/Work/slam-super-6d'; % slam-super-6d dir
% result_dir = horzcat(root_dir, '/experiments/ycbv/inference/003_cracker_box_16k/ycb_gt');
% result_dir = horzcat(root_dir, '/experiments/ycbv/dets/ground_truth/003_cracker_box_16k');
result_dir = horzcat(root_dir, '/experiments/ycbv/dets/ground_truth/003_cracker_box_16k'); % set path to results
test_cls_idx = 2; % 1-indexed (e.g. 2 = cracker box, 3 = sugar, 9 = meat can)

% read dope2ycb fixed transformation
json_file = horzcat(root_dir, '/experiments/ycbv/_ycb_original.json');
fid = fopen(json_file);
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
json_data = jsondecode(str);
dope2ycb_trsfm = json_data.exported_objects(test_cls_idx).fixed_model_transform';
dope2ycb_trsfm(1:3, :) = dope2ycb_trsfm(1:3, :) / 100;
dope2ycb_trsfm = dope2ycb_trsfm(1:3, 1:3);

% read the result directory and preprocess
all_seqs = dir(horzcat(result_dir, '/*.txt'));
for s = 1:numel(all_seqs)
    seq_file = horzcat(all_seqs(s).folder, '/', all_seqs(s).name);
    content = readmatrix(seq_file);
    [~,all_seqs(s).seq_id,~] = fileparts(all_seqs(s).name);
    all_seqs(s).seq_id = all_seqs(s).seq_id(1:4); % in case redundancy appended
    all_seqs(s).frame_ids = arrayfun(@(id) sprintf('%06d', id), ...
        content(:,1) * 10 + 1, 'UniformOutput', false); % cell (n,1)
    all_seqs(s).miss_detections = arrayfun(@(row) check_miss_detection(content(row, 2:8)), ...
        (1:size(content,1)))'; % (n,1) logical in {0,1} transpose makes vertical
    all_seqs(s).translations = arrayfun(@(row) content(row, 2:4)', ...
        (1:size(content,1))', 'UniformOutput', false); % cell (n,1) (3,1)
    all_seqs(s).rotations = arrayfun(@(row) quat2rotm(horzcat(content(row,8), ...
        content(row, 5:7))) * dope2ycb_trsfm, (1:size(content,1))', ...
        'UniformOutput', false); % cell (n,1) (3,3) quat(xyzw)->quat(wxyz) in matlab
%     all_seqs(s).poses = arrayfun(@(row)  vertcat(horzcat(quat2rotm(horzcat(content(row,8), content(row, 5:7))), content(row, 2:4)'), [0 0 0 1]) * dope2ycb_trsfm, ...
%         1:size(content,1), 'UniformOutput', false); % cell (n,1) (4,4). Change 3 places
end

% read class names
fid = fopen('classes.txt', 'r');
C = textscan(fid, '%s');
object_names = C{1}; % (21,1) cell
fclose(fid);

% load model points
num_objects = numel(object_names);
models = cell(num_objects, 1);
for i = 1:num_objects
    filename = fullfile(opt.root, 'models', object_names{i}, 'points.xyz');
    disp(filename);
    models{i} = load(filename); % 3d point clouds (n x 3 mat)
end

% load the keyframe indexes
fid = fopen('keyframe.txt', 'r');
C = textscan(fid, '%s');
keyframes = C{1}; % n x 1 cell of strings
fclose(fid);

% save results
distances_sys = zeros(100000, 5);
distances_non = zeros(100000, 5);
errors_rotation = zeros(100000, 5); 
errors_translation = zeros(100000, 5);
results_seq_id = zeros(100000, 1);
results_frame_id = zeros(100000, 1);
results_object_id = zeros(100000, 1);
results_cls_id = zeros(100000, 1);

% for each image
count = 0;
for i = 1:numel(keyframes)
    
    % parse keyframe name
    name = keyframes{i};
    pos = strfind(name, '/'); % find position of / in seq/frame_id
    seq_id_str = name(1:pos-1);
    seq_id = str2double(name(1:pos-1));
    frame_id_str = name(pos+1:end);
    frame_id = str2double(name(pos+1:end));
    
    % load gt
    filename = fullfile(opt.root, 'data', sprintf('%04d/%06d-meta.mat', seq_id, frame_id));
    gt = load(filename);
    
    for j = 1:numel(gt.cls_indexes) % 1-indexed (see plot_accuracy_keyframe.m/L32
        if gt.cls_indexes(j) == test_cls_idx
            disp(filename);
            count = count + 1;
            cls_index = gt.cls_indexes(j);
            results_seq_id(count) = seq_id;
            results_frame_id(count) = frame_id;
            results_object_id(count) = j;
            results_cls_id(count) = cls_index; % vectors
            
            % locate seq_id and frame_id in results
            for s = 1:numel(all_seqs)
                if strcmp(all_seqs(s).seq_id, seq_id_str)
                    break;
                end
            end
            %disp(all_seqs(s).seq_id)
            %disp(seq_id_str)
            assert(strcmp(all_seqs(s).seq_id, seq_id_str), ...
                'possible cause: object appears in a seq in the ground-truth but that seq is not in results');
            frame_id = round(frame_id);
            assert(strcmp(all_seqs(s).frame_ids{frame_id}, frame_id_str));
            
            % check miss detection
            if all_seqs(s).miss_detections(frame_id)
                fprintf('miss detection in seq %s fr %s\n', all_seqs(s).seq_id, all_seqs(s).frame_ids{frame_id}); 
                distances_sys(count, 1) = inf;
                distances_non(count, 1) = inf;
                errors_rotation(count, 1) = inf;
                errors_translation(count, 1) = inf;
            else
                % get poses
                RT_gt = gt.poses(:,:,j);
                RT = zeros(3, 4);
                RT(1:3, 1:3) = all_seqs(s).rotations{frame_id};
                RT(:, 4) = all_seqs(s).translations{frame_id};
                % RT = all_seqs(s).poses{frame_id}(1:3, :);

                % evaluate
                distances_sys(count, 1) = adi(RT, RT_gt, models{cls_index}');
                distances_non(count, 1) = add(RT, RT_gt, models{cls_index}');
                errors_rotation(count, 1) = re(RT(1:3, 1:3), RT_gt(1:3, 1:3));
                errors_translation(count, 1) = te(RT(:, 4), RT_gt(:, 4));
            end
        end
    end
end

distances_sys = distances_sys(1:count, :); % to remove extra zeros
distances_non = distances_non(1:count, :);
errors_rotation = errors_rotation(1:count, :);
errors_translation = errors_translation(1:count, :);
results_seq_id = results_seq_id(1:count);
results_frame_id = results_frame_id(1:count);
results_object_id = results_object_id(1:count, :);
results_cls_id = results_cls_id(1:count, :);
save('results_keyframe.mat', 'distances_sys', 'distances_non', 'errors_rotation', 'errors_translation',...
    'results_seq_id', 'results_frame_id', 'results_object_id', 'results_cls_id');

function is_miss_detection = check_miss_detection(input_row)
if isequal(input_row, [0 0 0 0 0 0 0])
    is_miss_detection = true;
else
    is_miss_detection = false;
end
end

function pts_new = transform_pts_Rt(pts, RT)
%     """
%     Applies a rigid transformation to 3D points.
% 
%     :param pts: nx3 ndarray with 3D points.
%     :param R: 3x3 rotation matrix.
%     :param t: 3x1 translation vector.
%     :return: nx3 ndarray with transformed 3D points.
%     """
n = size(pts, 2);
pts_new = RT * [pts; ones(1, n)];
end

function error = add(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with no indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);
diff = pts_est - pts_gt;
error = mean(sqrt(sum(diff.^2, 1)));
end

function error = adi(RT_est, RT_gt, pts)
%     """
%     Average Distance of Model Points for objects with indistinguishable views
%     - by Hinterstoisser et al. (ACCV 2012).
% 
%     :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
%     :param model: Object model given by a dictionary where item 'pts'
%     is nx3 ndarray with 3D model points.
%     :return: Error of pose_est w.r.t. pose_gt.
%     """
pts_est = transform_pts_Rt(pts, RT_est);
pts_gt = transform_pts_Rt(pts, RT_gt);

% Calculate distances to the nearest neighbors from pts_gt to pts_est
MdlKDT = KDTreeSearcher(pts_est');
[~, D] = knnsearch(MdlKDT, pts_gt');
error = mean(D);
end

function error = re(R_est, R_gt)
%     """
%     Rotational Error.
% 
%     :param R_est: Rotational element of the estimated pose (3x1 vector).
%     :param R_gt: Rotational element of the ground truth pose (3x1 vector).
%     :return: Error of t_est w.r.t. t_gt.
%     """

error_cos = 0.5 * (trace(R_est * inv(R_gt)) - 1.0);
error_cos = min(1.0, max(-1.0, error_cos));
error = acos(error_cos);
error = 180.0 * error / pi;
end

function error = te(t_est, t_gt)
% """
% Translational Error.
% 
% :param t_est: Translation element of the estimated pose (3x1 vector).
% :param t_gt: Translation element of the ground truth pose (3x1 vector).
% :return: Error of t_est w.r.t. t_gt.
% """
error = norm(t_gt - t_est);
end