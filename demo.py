import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, 'TPS', 'ResNet',
          'BiLSTM', 'Attn')
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            results_path = opt.saved_model.split('/')[-2]
            
            log = open(f'./results/{results_path}_result.txt', 'a')
            # import pdb; pdb.set_trace()
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=40, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ폽랜땜푬걀빠켓템펨쓰꿈흐윽쿠맴퍼연떻멘신짓폳윤쏫깍렴잇뤄샤즉쏘쌀펀상놈텔안듦튼원맡증멀순왯쟁텁같맘과왛빅젿홈윺퀘챔냥빚소석궁성섞크빡뻐치위메벨쫓탇됏언정하탄깥젲온느될숭삼쎄축딤쥐멪근단얄떡난깜즌욎윶퓨등엷외칙앗윱졋물흥늑름젖뺨앰료빵듬붐싹갖눈환딕규쇄찾야사넷몸덥톶솥뼉델톧뛰왔많턴까낼펌혼펍깡차표긋얇빔러춤고왐쩍출킻났쿤걔립얘쿱드좀쩜졍족험싣큼늦컨킬높먿왙타침윧얽혁쩐풉쿄멋닿베벗큰풋맿믿흑째질넌틈젤닭펠최조찻펼을꾼여겨괜괴반끗컵독햇팬럽픋협갱뜨앵씹트횡펙몇졈챙회라퉁개꺾구피겁렇적꽃쑤션각욜옵짧겪걱숀핸왜쩌귤몁폇훈건술뻔팅펩뭄잣뚫캠림도빛새앓칩송갵액책광쉽첫떼넉계섭잡픙의렉숙게윗브꼭랫탣젱보니텓펴참옹푿뜻켯바찍껏줍딱칠간망싼부십끓왈붕왱낮럿필혜꼼유몽략췌운낫썩깅밀앳국란훌락곧덧버덜녹만듀엇된룹이숟리댓점능클뜯못젊샌설층꽂욥헌디얼실푤권으킫금지튜념너옷겹졀제욪푭않직푣따뇌렷집틀총첨두뎀흉남킵팽뒤돼맣웃분병낯컷헬애털현므처칼왣낌톡쟤욕응였팩냇눕미멥프횟뵙받거퀴습돈닌잠곡방럼앱향람끊띄낻데및행세싱멸왝앤얀짙강륭페갈갓욍덤코졌죄츠꾸닷밝펻년욉별암백굶판벽토일룻콘쓴맬앉탱학무목둥볕밴투뺏쁨촌몀겅열흩떠덕낡서핑밟생혀른욘더찰워후탠육촬왼멜손재왑엌묘푝껌딧엿압군허닮퐃전귓합칭동농슬켰놀득머읽나늬탤봄텡딸쾌키앧쳐굵뿐닙식픕율래졸굳곤져택측은숨감펒종닝편놓푲숫령멉웅죽쉰쭉혐폼릇쓸먹킹갣번그봇취폴밉삶한릭컬닦효녀슴젠촉맺푹젹낱냅훨빝딘장없섹탯뜰슷말폭매당룸포톨청루든멎톤널좋려넣펵낚짚띠짝쩔푱숄쏜중김왕왬배되욈문톳와빋빈옥몬산킷음체뽑기님케악본때스굉약며킴추주렁법쉬커객폐캄써쏨쨌노업통뒷멧함봉탐갭폅깨왚꽤붉왭영텀듭를왁테쭈색르셈천늘젓히률얹롱얗빼셔갷엎뭘싶날들솔올플돗익멈흡호뱅비답확쏠수했욋다빙극퐁썰숲컴춥로끈씨엉홍쌓덩쪽춰딪맞먼늄갤논캐옫룩콜왇킥꽉뚜셋랑왤텍탓꼽넥채알녕좁멍즈씩자풍승냐늙아듣둡됫밖살몹곳텟옮특탑갇잉쿵퐅깊공뉴릎푼벌꿀격좌할볍씀핵글찬벼뼈왓탁선철촛튿불맙탕넓린뿌복팝형시롬롭섯억밥됐랩움량갛레껑견초껴낭똑덮맛묶킨웨텃저쿼풎젣혹갯꼬픔력룰춧속숍빗끼엊졉멩곁덱뜩왹픗갠끄오텅꼴빞짜밑읻폿뭬검젭줌긁긴돕엄엔던윷퐇맨담항쏟둘깔턷뷰슈막파듄픈패냄낳쁘헨쌍갑엽골앨욀카펑붓뇨뮤혔잦닉쇠싸팎싯퇴젬렬끝해용잊활텐내가콩깎잃에젯걸옴모걷빨준빕즐씌펄균대탈왘진박됨창밌힘릿잔녁렵젝볼변찌땅갹춘둑넘관발록폰런뱃옳줄뀌흙멤평밋맑푸울밍덴엘찮멛었뢰퍽옆씬풀또결둠뵈몃눌맷잘삭즘첩롣달쇼획입왗곰욱센갲맥됩혓펟덟콤멷범경끌황갳퐂어푠혈뱀급갚짐윰얕맻왖뱉벤척뻗염썹징딩훔햄툴티잎꺼랗교례냉뉘납떤겸윳칫팔텆딴댐블는찢쿰럴헤접뭣깝갼께깐홉캡랙갸우옛픽희품탬닫뭐딛낸씻면랍폄껍화싫턱킺듯밭충슨류웬휘곽착켜요값융터왠볶았절존명붙푯론룬롤낙끔완탭홀잖갶롯딜앞펃펏컫휴룽누렌땀댁멕젼핏좃뉜련묻뭇빌펫굴겉링귀것섬떨몫흔펜있월툼낵랭인태작곱북팀마럭괄얻믹예돌놔닥됭양츰륙솟긍역민튀친틱븐맹뚱맵삿네쿨욧묵욛톱닐굽흘밤심흰칸핀임릴옿램길맫앙몰솜묄샷끎힐섈쏴슁녘혠뿅뿔챌닸빎떽쥴뜅넙쾀늪횰잗콕혤낄뇜뮨슘엥괏뎐묠붸딨띌쯧껙꽈죠겟넜뜁뗬칟꿰뤽홋죡촨켐옇휫쬈츄쐈듈챤퓜뮌죗쐐깬굡씸쁜곈뒹궉똘띈헵궜굣븃벧뽈뢴꽁뺌랐뢨륨쁩힙훼즛섀뜹꼇췻풂푄넨쭹죕먈퍅뷴갔댕퐈퐝걍궂썼섦툿찝빪췬겻뎄퉜컁뱍붤껄팟훗췰삯쌌샬묽쟀훅뇟꺽챘콧깠찡꿜쾅슝벡쨉늣깩뱐굻좇뇐곬꾈팁쏸촐깃쉴돠쉔훰꽹흴솰윅켸봐뷜삳됴놉봅갰릊쫬쇗륀륏뵨팼좍삵넒퀼눙켱뾔꾜섟뎬뎔뗄옌쿡벴쳅켄쪘힉냘샜툐뉼츌퉤냠폘랠띱솽쫀낀롓벱돐늠씜룡쟐틸켁뗐휸홴폡쬡퀄슉뇹탔휵쥠뭔좝죤뵌쑹펐힛꼍걺쇨퓐쒜훽쮸쵤퓸뼙벵썅퓰뤼곗팜겠샵팸둔샀솨긷쥑똬뤘엡졺쇈깽뎃휭똔켈읍륩퓌좆펭끅캔굄밸몲쳔눼묑삑쨀닳뒬롄뒝삡쯔귿윈훠홧톼샹챦댈돎헝챠씁벎볜삥쳄걘텨쇘떴잰훑묾뀨옅뼁귁놜띔뙤옰뎅뉵뵉뺘괘짇샥숴웰끽잚웍뫙긱쌘솬읠쫌죵잴괼엠큔줴뾰뵘봤겡훙쎌깆뇻쥘닺떫땍쐼읒띨췸먀틤흗샙껜뼜튁쟨땐겊뚠큘닛놂셤훵궝찼셍돝줬홰뛔귄퇀껼챈뇝쉼짭뉨꺌촁뭉떵짰뽄뜸밂껸딥쯤뗀꾀냔뉩뀁꽜괭쌜츨튄쨈큽쮜삘꼰룟붰챨췹묀쾰욤띵짊쫏똴쟝셀꾹톺옐뚤뮴뀄쑥뽀녔뎡컸룀궈첸쉠겋뎁듕뭡깼턍캅뻣괍쨍닒눔섧휨룐퀑샨뒵겆벳쬐땔멱켭땄붊캭뻤퓟틥튬찐켠웝읫퀀웸렘홑믓겯헙띳랏핍꿱쯩쫴뇩헐쳇흇곪쫙믈틋덛냑뼛눋읨닯쩡쭝솎덫쉐쇔둬탉먕뻥뫄훤캬셕쌥쳤잽쁠첵좔횔벚횐뺙좟헴웁읗넸뿡쵭긔옻섄쬠굔큄촙랴숱뗘셴뢍맸숯퓻횻짢쏢쩨늚뫼녜떪짠팡쐽샴쩽숩꿴덖륄쫘꾐묫튱뺐봬쉈횃굿짬햐뒨쟘콰틴껫큭틂흼엑칡씽틉셸캉갊뇬웩첼뇔쵬쉑쟌쒼삐쵸쇳퇸횬쬘빳팹눅줘튠괩엾짖뮐캇굼힝퍄쟬쫠늅죙셨읏괸쇤랄깁톄겐찹륌묩빱겔꽐켕쬔팥셥쭁넛쇰떱몄뜀쒔힌괵뮈뺍뿟샐엮쥬쭙럇켑넴떳쉘뉠헉뀐뒈밞횝깟귑흖콱꽝룔븅햅읓늉낟렝츔묏싻씔쳰녈팍깹븍괬겄밗쁑먁껨썽탰굘얏흄듐윕슐갬썸옘췽쏩촘뿜넹쳬켤풩팰쓩뺄뙨띰떰쿳쪄챗퓔꿉꾑졔핼휑킁챰뮷돤샘녠툇쵯욺헹얜깻촹챵쩠뻘넋윌붜좼꿋돛뭏셜츳솝웽껭늴쟈툉뤠웠퀭빰쨔릅뗌놘쨋핥팖릉뎨쐬숏곶읜묍걜귈컹쭸쑈풔뫈꺄짯쭐앎숌쒸벰휩섐읖큇띤잿뇽잭쇌툭룅죔띕뻠텬쳉붇쌔븜귐헷솩뭅뿍릍꼲캘쾨씰쓿튑섰꺅폣뱁씐얠꼿쓱쇽땝쑴뵐혭겼렙쌕욹퀸쥼몌헛췄줅렐큠얌읾휠삣숑붚켬짤촤뛴룃윔갉긺힁굅슛쐰퀵뜬롼춈쀼롸췐뷕꼐쉿쥰셌쭘앍휼뱌퓽댜틘틜셧꿸폈옙돨놋줏옜쭌콥흽흠찔뺑붑녑븀맏뀀쓺콴뛸꿩빻볘츤쟉엣뺀뀝뽕밈깰댄꿨웜닢틔꿇쌩웹쥔휀쵠쭤돋갗쉭븟꾕헥낢얩렛읔슥솖퓬흣튈셩윙땁꾄팠꿔칵랸쏙쩟뱄걋괌꽥넝둣껀쎈쌉뚝얾잤궐뵀톈푀뎌댔즙쥣끙짱뭍멓궤쨩껐콸뗍읊쒀핫뷸츱묜돔벅샅쑬벋뎠뼘뼝좡쵱븝룁헒튤뻬튕쌨뵤팻큅롑퉈뙈끕츈뢸괠렸멂놨툰맒웡캥빴벙빤쌤읕쌈앝쥡괆곯쨘낏슭큉쾡캣엶눗앴쫄삔쏵댑뒀뀔룝뚬넬쐤섶쏭빽덞솅땟륜짼멨륑셰냈뷔땃뱝쑵땡탸얍큐컥쉥땋휜늰곕쇱죈핌렀팃뇰눴땠닻톰봔뽐뽁굇곌뗑홅듸옭겜꿍쫍쏀찜깸뷩좨밧홱짹쪼겝좽샛꼈퇘랒쑨뗏휄믄텄찧눠쐴쫑썲셉늡첬랖튐뻑륵뀜뷘횅똥텼캑썬궷쩝삠낍휙삽횹랬캤턺텝꿎츙줆죌낑볏뫘믐넵깖섕볐툽륫잼쟎뮬섣챕?!,.*()[]-+=%<>&/;:#※○□·“「」}{_^$±¥£€₩',
                         help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
   
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)

# CUDA_VISIBLE_DEVICES=0 python3 demo.py --image_folder /home/user01/ocr_recognition/deep-text-recognition-benchmark/demo_image --saved_model /home/user01/ocr_recognition/deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-Attn-Seed11118/best_accuracy.pth --rgb --PAD