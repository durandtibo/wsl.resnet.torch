
local paths = require 'paths'

-- path where download models
local path_pretrained_model = 'data/pretrained_models'

paths.mkdir(path_pretrained_model)

print('Download pretrained models in ' .. path_pretrained_model)

-- ResNet-18
print('Download ResNet-18...')
local filename = path_pretrained_model .. '/resnet-18.t7'
if not paths.filep(filename) then
    local url = 'https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7'
    os.execute('wget ' .. url .. ' -O ' .. filename)
end
print('Done\n')

-- ResNet-101
print('Download ResNet-101...')
local filename = path_pretrained_model .. '/resnet-101.t7'
if not paths.filep(filename) then
    local url = 'https://d2j0dndfm35trm.cloudfront.net/resnet-101.t7'
    os.execute('wget ' .. url .. ' -O ' .. filename)
end
print('Done\n')
