%你的努力在未来的一天都会回来找你的-2020.1.22


%参数设置
base_path = 'D:\mysoft_download\CSK_tracker_release\data\';
%% 1.choose_video函数解释
%==========================================================================
%==========================================================================
%输入  base_path
%输出  video_path
%功能  可以选择想要测试的数据集，得到想要的数据集的路径，方便后面调用
%video_path = choose_video(base_path)

if ispc(),base_path = strrep(base_path,'\','/');end
if base_path(end) ~= '/',base_path(end+1) = '/';end

contents = dir(base_path);
names = {};
%读取data文件夹里的每一个类型的数据集
for k=1:numel(contents)
    name =contents(k).name;
    if isfolder([base_path name])&& (~strcmp(name,'.'))&&(~strcmp(name,'..'))
        names{end+1} = name;
    end
end

%如果在base_path下没有子文件夹，直接使用return命令退出
if isempty(names),return;end
%choice 返回的是索引
choice = listdlg('ListString',names,'Name','choose a video','SelectionMode','single');

if isempty(choice)
    video_path =[];
else
    %字符串使用[]连接
    video_path = [base_path names{choice} '/'];
end
%video_path准备就绪
%==========================================================================
%==========================================================================
%%
%如果前面在choose_video 环节没有选到video 就可以退出了
if isempty(video_path),return;end
%% 2.load_video_info代码解释
%==========================================================================
%==========================================================================
% [img_files, pos, target_sz, resize_image, ground_truth, video_path] = ...
% 	load_video_info(video_path);
%输入 前面选择的video_path 
%输出 img_files 输出的图片名，类型为struct
%输出 pos 初始帧的中心位置 ，（sy，sz）
%输出 target_sz 初始帧框的大小，(height,width)
%输出 resize_image 图片是否缩放了，有时框太大需要降低分辨率
%输出 ground_truth 每一帧所对应的ground_truth，[x,y,width,height]
%输出 video_path 在前面video_path的基础上 进入到/img子文件夹


%选出video_path路径下以.txt结尾的文件
%这样就不用管dir可能输出'.'和'..'了
text_files = dir([video_path '*.txt']);
assert(~isempty(text_files),['??? no groundtruth.txt in the ' video_path])  

f = fopen([video_path text_files(1).name]);
%textscan读出来是cell
ground_truth = textscan(f,'%f,%f,%f,%f');%[x,y,width,height]
ground_truth = cat(2,ground_truth{:});
fclose(f);

%设置初始位置和大小
%target_sz(1)是height 对应 y
%target_sz(2)是width 对应 x
%图片左上角为坐标零点
%（x,y)是左上角坐标，（x+width，y+height）是右下角坐标
target_sz = [ground_truth(1,4),ground_truth(1,3)];
pos =[ground_truth(1,2),ground_truth(1,1)]+floor(target_sz/2);

video_path = [video_path 'img/'];
%.png和.jpg格式的文件都可以读取
img_files = dir([video_path '*.png']);
if isempty(img_files)
    img_files =dir([video_path '*.jpg']);
    assert(~isempty(img_files),'no jpg or png to be read')
end
img_files = sort({img_files.name});

%如果目标太大，使用低一点的分辨率
if sqrt(prod(target_sz))>=100
    pos = floor(pos/2);
    target_sz = floor(target_sz/2);
    resize_image =true;
else
    resize_image = false;
end

%img_files, pos, target_sz, resize_image, ground_truth, video_path准备就绪
%==========================================================================
%==========================================================================
%% 3.论文给的参数
padding = 1;					%extra area surrounding the target
output_sigma_factor = 1/16;		%spatial bandwidth (proportional to target)
sigma = 0.2;					%gaussian kernel bandwidth
lambda = 1e-2;					%regularization
interp_factor = 0.075;			%linear interpolation factor for adaptation
%% 4.准备高斯响应图和cosine窗
%由target_sz扩大范围为sz
sz =floor(target_sz*(1+padding));
%gaussion 参数
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor;
%就是类似于网格化计算吧
%rs 和 cs的size是sz(1)*sz(2)
[rs,cs]=ndgrid((1:sz(1))-floor(sz(1)/2),(1:sz(2))-floor(sz(2)/2));
%y的size是sz(1)*sz(2)
y =exp(-0.5/output_sigma^2*(rs.^2+cs.^2));
%对高斯响应图进行fft
yf=fft2(y);

%储存cosine window
cos_window = hann(sz(1))*hann(sz(2))';
%% 5.开始训练和跟踪流程
time = 0;%计算FPS
positions = zeros(numel(img_files),2);%记录每次定位的位置，用来计算precision

for frame = 1:numel(img_files)
   %5.1读取image
   im = imread([video_path img_files{frame}]);
   if size(im,3)>1
       im = rgb2gray(im);
   end
   
   if resize_image 
       im = imresize(im,0.5);%在第2小节中有提及resize_image的来历
   end
   
   tic()
   %5.2提取以及预处理子窗
   x = get_subwindow2(im,pos,sz,cos_window);
   
   %5.3计算最大响应
   if frame >1 %不是第一帧的话需要检测出新位置后更新
       %计算分类器(z)在所有位置(x)的响应
       %找到最大的位置
       k =dense_gauss_kernel2(sigma,x,z);
       response = real(ifft2(alphaf.*fft2(k)));%eq.9
       %在响应response最大的位置定位出新的中心位置
       [row,col] = find(response == max(response(:)),1);
       %更新pos
       pos = pos - floor(sz/2)+[row,col];
       
   end
   
   %除了第一帧，在更新完pos之后，获取新位置下的子窗
   %5.4训练分类器，也就是更新alphaf
   x = get_subwindow2(im,pos,sz,cos_window);
   
   %Kernel Regularized Least-Squares，在傅里叶域计算alphas
   k = dense_gauss_kernel2(sigma,x);
   
   %基于x计算k，基于k计算new_alphaf
   new_alphaf = yf./(fft2(k)+lambda);
   %new_z 就等于更新完的pos，新位置的子窗
   new_z =x;
   
   %5.5更新机制
   if frame ==1
       alphaf = new_alphaf;
       z = new_z ;
   else
       %引入学习率机制
       alphaf = interp_factor * new_alphaf + (1-interp_factor)*alphaf;
       z = interp_factor *new_z +(1-interp_factor) * z;
   end
    
        
   
   %保存位置，计算FPS
   %把postions换成【x，y】形式，方便后面算与ground_truth 【x,y,width,height】
   %的距离
   positions(frame,:)=pos([2,1])-floor(target_sz([2,1])/2);
   time =time +toc();
   
   %5.6可视化
   %修正之后的[x,y,width,height]
   rect_position = [pos([2,1])-target_sz([2,1])/2,target_sz([2,1])];
   if frame == 1%第一帧，创建GUI
       figure('UserData','off','Name',['Tracker - ' video_path]);
       im_handle = imshow(im,'Border','tight','InitialMag',200);
       rect_handle = rectangle('Position',rect_position,'EdgeColor','g');
   else
       %更新GUI
       set(im_handle,'CData',im);
       set(rect_handle,'Position',rect_position);
       
       
       
   end
   
   drawnow
   
   
   
end
%% 善后工作，resize相关，fps相关，precision相关
if resize_image,positions = positions *2 ; end
disp(['Frames-per-second: ' num2str(numel(img_files) / time)])
%show the precisions plot
show_precision2(positions, ground_truth, video_path)
%% 调用的子函数 get_subwindow2
function out = get_subwindow2(im,pos,sz,cos_window)
    %判断sz输入是不是一个标量
    if isscalar(sz)
        sz = [sz,sz]
    end
    %得到网格化的x和y坐标
    ys = floor(pos(1))+(1:sz(1))-floor(sz(1)/2);
    xs = floor(pos(2))+(1:sz(2))-floor(sz(2)/2);
    
    %防止出边界
    xs(xs<1)=1;
    ys(ys<1)=1;
    xs(xs>size(im,2))=size(im,2);
    ys(ys>size(im,1))=size(im,1);
    
    %提取出子窗
    out = im(ys,xs,:);
    
    %对子窗进行预处理
    out = double(out)/255-0.5;%归一化到-0.5到0.5
    out = cos_window.*out;%运用cosine窗


end

%% 调用的子函数 dense_gauss_kernel2
function k = dense_gauss_kernel2(sigma,x,y)
    
    %x的傅里叶变换
    xf = fft2(x);
    %x的平方范数，xx是一个数
    % x(:)将 x展成了 size(x,1)*size(x,2) X 1的列向量
    xx = x(:)'*x(:);
    
    %x,y不同，输入参数为3
    if nargin >= 3
        yf = fft2(y);
        yy=y(:)'*x(:);
    else
        %也就是输入为2，x与x自相关
        yf = xf;
        yy = xx;
    end
    %傅里叶域的互相关形式
    xyf = xf .*conj(yf);
    %回到时域，为什么这里要使用 circshift去移动？
    xy = real(circshift(ifft2(xyf),floor(size(x)/2)));
    
    k =  exp(-1/sigma^2*max(0,(xx+yy-2*xy)/numel(x)));
    
        

end

%% 调用的子函数 show_precision2
function show_precision2(positions,ground_truth,title)
    max_threshold = 50;  %文章中用到的最大
    
    if size(positions,1)~=size(ground_truth,1)%比较两个变量对应帧数是否相同
        disp('Could not plot precisions, because the number of ground')
        disp('truth frames does not match the number of tracked frames.')
        return
    end
    
    %计算positions在每一帧与ground_truth之间的距离
    distances =sqrt((positions(:,1)-ground_truth(:,1)).^2+...
    (positions(:,2)-ground_truth(:,2)).^2);
    distances(isnan(distances))=[];
    %计算精确度
    precisions = zeros(max_threshold, 1);
	for p = 1:max_threshold
        %计算所有帧中距离小于p的帧所占比例
		precisions(p) = nnz(distances < p) / numel(distances);
	end
	
	%plot the precisions
	figure('UserData','off', 'Name',['Precisions - ' title])
	plot(precisions, 'k-', 'LineWidth',2)
	xlabel('Threshold'), ylabel('Precision')
end

















    