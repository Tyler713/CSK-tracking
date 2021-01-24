%���Ŭ����δ����һ�춼����������-2020.1.22


%��������
base_path = 'D:\mysoft_download\CSK_tracker_release\data\';
%% 1.choose_video��������
%==========================================================================
%==========================================================================
%����  base_path
%���  video_path
%����  ����ѡ����Ҫ���Ե����ݼ����õ���Ҫ�����ݼ���·��������������
%video_path = choose_video(base_path)

if ispc(),base_path = strrep(base_path,'\','/');end
if base_path(end) ~= '/',base_path(end+1) = '/';end

contents = dir(base_path);
names = {};
%��ȡdata�ļ������ÿһ�����͵����ݼ�
for k=1:numel(contents)
    name =contents(k).name;
    if isfolder([base_path name])&& (~strcmp(name,'.'))&&(~strcmp(name,'..'))
        names{end+1} = name;
    end
end

%�����base_path��û�����ļ��У�ֱ��ʹ��return�����˳�
if isempty(names),return;end
%choice ���ص�������
choice = listdlg('ListString',names,'Name','choose a video','SelectionMode','single');

if isempty(choice)
    video_path =[];
else
    %�ַ���ʹ��[]����
    video_path = [base_path names{choice} '/'];
end
%video_path׼������
%==========================================================================
%==========================================================================
%%
%���ǰ����choose_video ����û��ѡ��video �Ϳ����˳���
if isempty(video_path),return;end
%% 2.load_video_info�������
%==========================================================================
%==========================================================================
% [img_files, pos, target_sz, resize_image, ground_truth, video_path] = ...
% 	load_video_info(video_path);
%���� ǰ��ѡ���video_path 
%��� img_files �����ͼƬ��������Ϊstruct
%��� pos ��ʼ֡������λ�� ����sy��sz��
%��� target_sz ��ʼ֡��Ĵ�С��(height,width)
%��� resize_image ͼƬ�Ƿ������ˣ���ʱ��̫����Ҫ���ͷֱ���
%��� ground_truth ÿһ֡����Ӧ��ground_truth��[x,y,width,height]
%��� video_path ��ǰ��video_path�Ļ����� ���뵽/img���ļ���


%ѡ��video_path·������.txt��β���ļ�
%�����Ͳ��ù�dir�������'.'��'..'��
text_files = dir([video_path '*.txt']);
assert(~isempty(text_files),['??? no groundtruth.txt in the ' video_path])  

f = fopen([video_path text_files(1).name]);
%textscan��������cell
ground_truth = textscan(f,'%f,%f,%f,%f');%[x,y,width,height]
ground_truth = cat(2,ground_truth{:});
fclose(f);

%���ó�ʼλ�úʹ�С
%target_sz(1)��height ��Ӧ y
%target_sz(2)��width ��Ӧ x
%ͼƬ���Ͻ�Ϊ�������
%��x,y)�����Ͻ����꣬��x+width��y+height�������½�����
target_sz = [ground_truth(1,4),ground_truth(1,3)];
pos =[ground_truth(1,2),ground_truth(1,1)]+floor(target_sz/2);

video_path = [video_path 'img/'];
%.png��.jpg��ʽ���ļ������Զ�ȡ
img_files = dir([video_path '*.png']);
if isempty(img_files)
    img_files =dir([video_path '*.jpg']);
    assert(~isempty(img_files),'no jpg or png to be read')
end
img_files = sort({img_files.name});

%���Ŀ��̫��ʹ�õ�һ��ķֱ���
if sqrt(prod(target_sz))>=100
    pos = floor(pos/2);
    target_sz = floor(target_sz/2);
    resize_image =true;
else
    resize_image = false;
end

%img_files, pos, target_sz, resize_image, ground_truth, video_path׼������
%==========================================================================
%==========================================================================
%% 3.���ĸ��Ĳ���
padding = 1;					%extra area surrounding the target
output_sigma_factor = 1/16;		%spatial bandwidth (proportional to target)
sigma = 0.2;					%gaussian kernel bandwidth
lambda = 1e-2;					%regularization
interp_factor = 0.075;			%linear interpolation factor for adaptation
%% 4.׼����˹��Ӧͼ��cosine��
%��target_sz����ΧΪsz
sz =floor(target_sz*(1+padding));
%gaussion ����
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor;
%�������������񻯼����
%rs �� cs��size��sz(1)*sz(2)
[rs,cs]=ndgrid((1:sz(1))-floor(sz(1)/2),(1:sz(2))-floor(sz(2)/2));
%y��size��sz(1)*sz(2)
y =exp(-0.5/output_sigma^2*(rs.^2+cs.^2));
%�Ը�˹��Ӧͼ����fft
yf=fft2(y);

%����cosine window
cos_window = hann(sz(1))*hann(sz(2))';
%% 5.��ʼѵ���͸�������
time = 0;%����FPS
positions = zeros(numel(img_files),2);%��¼ÿ�ζ�λ��λ�ã���������precision

for frame = 1:numel(img_files)
   %5.1��ȡimage
   im = imread([video_path img_files{frame}]);
   if size(im,3)>1
       im = rgb2gray(im);
   end
   
   if resize_image 
       im = imresize(im,0.5);%�ڵ�2С�������ἰresize_image������
   end
   
   tic()
   %5.2��ȡ�Լ�Ԥ�����Ӵ�
   x = get_subwindow2(im,pos,sz,cos_window);
   
   %5.3���������Ӧ
   if frame >1 %���ǵ�һ֡�Ļ���Ҫ������λ�ú����
       %���������(z)������λ��(x)����Ӧ
       %�ҵ�����λ��
       k =dense_gauss_kernel2(sigma,x,z);
       response = real(ifft2(alphaf.*fft2(k)));%eq.9
       %����Ӧresponse����λ�ö�λ���µ�����λ��
       [row,col] = find(response == max(response(:)),1);
       %����pos
       pos = pos - floor(sz/2)+[row,col];
       
   end
   
   %���˵�һ֡���ڸ�����pos֮�󣬻�ȡ��λ���µ��Ӵ�
   %5.4ѵ����������Ҳ���Ǹ���alphaf
   x = get_subwindow2(im,pos,sz,cos_window);
   
   %Kernel Regularized Least-Squares���ڸ���Ҷ�����alphas
   k = dense_gauss_kernel2(sigma,x);
   
   %����x����k������k����new_alphaf
   new_alphaf = yf./(fft2(k)+lambda);
   %new_z �͵��ڸ������pos����λ�õ��Ӵ�
   new_z =x;
   
   %5.5���»���
   if frame ==1
       alphaf = new_alphaf;
       z = new_z ;
   else
       %����ѧϰ�ʻ���
       alphaf = interp_factor * new_alphaf + (1-interp_factor)*alphaf;
       z = interp_factor *new_z +(1-interp_factor) * z;
   end
    
        
   
   %����λ�ã�����FPS
   %��postions���ɡ�x��y����ʽ�������������ground_truth ��x,y,width,height��
   %�ľ���
   positions(frame,:)=pos([2,1])-floor(target_sz([2,1])/2);
   time =time +toc();
   
   %5.6���ӻ�
   %����֮���[x,y,width,height]
   rect_position = [pos([2,1])-target_sz([2,1])/2,target_sz([2,1])];
   if frame == 1%��һ֡������GUI
       figure('UserData','off','Name',['Tracker - ' video_path]);
       im_handle = imshow(im,'Border','tight','InitialMag',200);
       rect_handle = rectangle('Position',rect_position,'EdgeColor','g');
   else
       %����GUI
       set(im_handle,'CData',im);
       set(rect_handle,'Position',rect_position);
       
       
       
   end
   
   drawnow
   
   
   
end
%% �ƺ�����resize��أ�fps��أ�precision���
if resize_image,positions = positions *2 ; end
disp(['Frames-per-second: ' num2str(numel(img_files) / time)])
%show the precisions plot
show_precision2(positions, ground_truth, video_path)
%% ���õ��Ӻ��� get_subwindow2
function out = get_subwindow2(im,pos,sz,cos_window)
    %�ж�sz�����ǲ���һ������
    if isscalar(sz)
        sz = [sz,sz]
    end
    %�õ����񻯵�x��y����
    ys = floor(pos(1))+(1:sz(1))-floor(sz(1)/2);
    xs = floor(pos(2))+(1:sz(2))-floor(sz(2)/2);
    
    %��ֹ���߽�
    xs(xs<1)=1;
    ys(ys<1)=1;
    xs(xs>size(im,2))=size(im,2);
    ys(ys>size(im,1))=size(im,1);
    
    %��ȡ���Ӵ�
    out = im(ys,xs,:);
    
    %���Ӵ�����Ԥ����
    out = double(out)/255-0.5;%��һ����-0.5��0.5
    out = cos_window.*out;%����cosine��


end

%% ���õ��Ӻ��� dense_gauss_kernel2
function k = dense_gauss_kernel2(sigma,x,y)
    
    %x�ĸ���Ҷ�任
    xf = fft2(x);
    %x��ƽ��������xx��һ����
    % x(:)�� xչ���� size(x,1)*size(x,2) X 1��������
    xx = x(:)'*x(:);
    
    %x,y��ͬ���������Ϊ3
    if nargin >= 3
        yf = fft2(y);
        yy=y(:)'*x(:);
    else
        %Ҳ��������Ϊ2��x��x�����
        yf = xf;
        yy = xx;
    end
    %����Ҷ��Ļ������ʽ
    xyf = xf .*conj(yf);
    %�ص�ʱ��Ϊʲô����Ҫʹ�� circshiftȥ�ƶ���
    xy = real(circshift(ifft2(xyf),floor(size(x)/2)));
    
    k =  exp(-1/sigma^2*max(0,(xx+yy-2*xy)/numel(x)));
    
        

end

%% ���õ��Ӻ��� show_precision2
function show_precision2(positions,ground_truth,title)
    max_threshold = 50;  %�������õ������
    
    if size(positions,1)~=size(ground_truth,1)%�Ƚ�����������Ӧ֡���Ƿ���ͬ
        disp('Could not plot precisions, because the number of ground')
        disp('truth frames does not match the number of tracked frames.')
        return
    end
    
    %����positions��ÿһ֡��ground_truth֮��ľ���
    distances =sqrt((positions(:,1)-ground_truth(:,1)).^2+...
    (positions(:,2)-ground_truth(:,2)).^2);
    distances(isnan(distances))=[];
    %���㾫ȷ��
    precisions = zeros(max_threshold, 1);
	for p = 1:max_threshold
        %��������֡�о���С��p��֡��ռ����
		precisions(p) = nnz(distances < p) / numel(distances);
	end
	
	%plot the precisions
	figure('UserData','off', 'Name',['Precisions - ' title])
	plot(precisions, 'k-', 'LineWidth',2)
	xlabel('Threshold'), ylabel('Precision')
end

















    