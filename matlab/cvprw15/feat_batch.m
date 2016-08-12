function [scores , list_im] = feat_batch(use_gpu, net_model, net_weights, list_im, dim)
caffe.reset_all();
if nargin < 1
  % By default use CPU
  use_gpu = 0;
end

if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
    caffe.set_mode_gpu();
    gpu_id = 0; % we will use the first gpu in this demo 
    caffe.set_device(gpu_id);
else 
    caffe.set_mode_gpu();
end 

% initialize the network using BVLC Caffet for image claffication
% weights(parameters) file needs to be downloaded from model zoo.
phase = 'test'; % run with phase test(so that dropout is not applied)

if ~exist(net_weights, 'file')
    error('%s doesnot exist.', net_weights);
end 
if ~exist(net_model, 'file')
    error('%s does not exist.', net_model);
end 

% Initialize a network 
net = caffe.Net(net_model, net_weights, phase);

% load mean file 
d = load('./matlab/+caffe/imagenet/ilsvrc_2012_mean.mat')
mean_data = d.mean_data;

% Adjust the batch size and dim to match with models/bvlc_reference_caffenet/deploy.prototxt
batch_size = 10;
disp(list_im)
if mod(length(list_im),batch_size) % if mod(len(.), batch_size)==0, not executes
    warning(['Assuming batches of ' num2str(batch_size) ' images rest will be filled with zeros'])
end

% prepare input
num_images = length(list_im);
scores = zeros(dim,num_images,'single');
num_batches = ceil(length(list_im)/batch_size);
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    input_data = {prepare_batch(list_im(range),mean_data,batch_size)};
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    output_data = net.forward(input_data);
    toc
    output_data = squeeze(output_data{1});
    scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
    toc(batchtic)
end
toc(initic);

% call caffe.reset_all() to reset caffe
caffe.reset_all();
end 
