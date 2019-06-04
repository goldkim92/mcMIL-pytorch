import torch

def get_instances(bag, patch_size, mc):
    '''
    bag : type:: torch.tensor
          size:: [1, channel, hight, width]
    mc : the number of instances to make, sampled from uniform distribution
    
    return : type:: torch.tensor
             size:: [mc, channel, patch_size, patch_size]
    '''
    _, _, bag_h, bag_w = bag.size()
    
    # `mc` number of top-left points sampled from uniform distribution
    point_randoms = torch.randint(0, (bag_h-patch_size)*(bag_w-patch_size), (mc,))
    t_randoms = point_randoms // (bag_w-patch_size)
    l_randoms = point_randoms % (bag_w-patch_size)
    tl_randoms = torch.stack([t_randoms, l_randoms], dim=1)
    
    # make instances
    instances = []
    for i in range(mc):
        tl = tl_randoms[i]
        instance = bag[:, :, tl[0]:tl[0]+patch_size, tl[1]:tl[1]+patch_size]
        instances.append(instance)
    instances = torch.cat(instances, dim=0)
    
    return instances


def get_instances_mesh(bag, ps, mc):
    _, _, bag_h, bag_w = bag.size()
    idxs_h, idxs_w = bag_h//ps, bag_w//ps

    # make instances in meshgrid style
    instances = []
    for idx_h in range(idxs_h):
        for idx_w in range(idxs_w):
            instance = bag[:, :, idx_h*ps:(idx_h+1)*ps, idx_w*ps:(idx_w+1)*ps]
            instances.append(instance)
    instances = torch.cat(instances, dim=0)

    return instances

