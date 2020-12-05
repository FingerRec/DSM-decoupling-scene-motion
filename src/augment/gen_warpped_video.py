from augment.forward_warp_python import Forward_Warp_Python
import torch


def gen_warpped_video(video, flows):
    fw = Forward_Warp_Python()
    gen_video = video.permute(0, 2, 1, 3, 4)
    flows = flows.permute(0, 2, 1, 3, 4)
    # print(video.size())
    for i in range(gen_video.size(0)):
        for j in range(gen_video.size(1) - 1):
            flow = torch.cat(([torch.unsqueeze(flows[i][2*j][0], 2), torch.unsqueeze(flows[i][2*j+1][0], 2)]), dim=2)
            flow = flow.unsqueeze(0)
            # print(flow.size())
            # print(flow.size())
            # print(video.size())
            # print(video.permute(0, 2, 1, 3, 4)[i][j].unsqueeze(0).size())
            gen_video[i][j+1] = fw.forward(video.permute(0, 2, 1, 3, 4)[i][j].unsqueeze(0), flow, 0).squeeze(0)
    gen_video = gen_video.permute(0, 2, 1, 3, 4)
    return gen_video