package recommender;
import recommender.common.*;
import recommender.common.RTuple;
// import recommender.common.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class SNLF extends CommonRec {

    public SNLF() {
        super();
    }

    public static void main(String[] args) throws IOException {
        int[] rsArr=new int[]{1,2,3,4,5};
        for(int rs: rsArr){
            CommonRec.dataSetName = "AT";
            CommonRec.rsDataSetName = String.valueOf(rs)+ "pos_"+ recommender.common.CommonRec.dataSetName;
            String filePath = "C:/Users/38457/Desktop/java/data/" + dataSetName + "/";
            CommonRec.dataLoad(CommonRec.dataSetName,filePath + CommonRec.rsDataSetName + "_train.txt",filePath + CommonRec.rsDataSetName + "_test.txt","::");
            System.out.println("当前物种的蛋白质总数："+ maxID);
            System.out.println("训练集的容量："+ CommonRec.trainDataSet.size());
            System.out.println("测试集的容量："+ CommonRec.testDataSet.size());
            


            CommonRec.maxRound = 1000;
            CommonRec.minGap = 1e-5;


            for(int tempdim = 100; tempdim <= 100;  tempdim += CommonRec.featureDimension){
                CommonRec.featureDimension = tempdim;
                CommonRec.featureSaveDir = "C:/Users/38457/Desktop/java/savedLFs/"+ CommonRec.dataSetName +"/A";
                CommonRec.biasSaveDir = "C:/Users/38457/Desktop/java/savedLFs/"+ CommonRec.dataSetName +"/B";
                // 初始化特征矩阵
                CommonRec.initiStaticFeatures();
                experimenter(false, CommonRec.RMSE);

            }
        }
    }

    public static void experimenter(boolean ifBias, int metrics) throws IOException {

        long file_tMills = System.currentTimeMillis(); //用于给train函数打开在当前函数所创建的文件
        FileWriter fw;
        if(metrics == CommonRec.RMSE)
            fw = new FileWriter(new File("C:/Users/38457/Desktop/java/feature/" + rsDataSetName + "_RMSE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "bias=" + ifBias  + "dim=" + featureDimension + ".txt"), true);
        else
            fw = new FileWriter(new File("C:/Users/38457/Desktop/java/feature/" + rsDataSetName + "_MAE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "bias=" + ifBias  + "dim=" + featureDimension + ".txt"), true);

        String blankStr = "                          ";
        String starStr = "****************************************************************";


        for (double tempLam = 0.008; tempLam >= 0.008; tempLam *= Math.pow(2,-1)) {

            CommonRec.lambda = tempLam;

            System.out.println("\n" + starStr);
            System.out.println(blankStr + "featureDimension——>" + CommonRec.featureDimension);
            System.out.println(blankStr + "lambda——>" + CommonRec.lambda);
            System.out.println(blankStr + "Bias——>" + ifBias);
            System.out.println(blankStr + "minGap——>" + CommonRec.minGap);
            System.out.println(starStr);

            fw.write("\n" + starStr + "\n");
            fw.write(blankStr + "featureDimension——>" + CommonRec.featureDimension + "\n");
            fw.write(blankStr + "lambda——>" + CommonRec.lambda + "\n");
            fw.write(blankStr + "Bias——>" + ifBias + "\n");
            fw.write(blankStr + "minGap——>" + CommonRec.minGap + "\n");
            fw.write(starStr + "\n");
            fw.flush();

            SNLF trainSNLF = new SNLF();

            // 开始训练
            trainSNLF.train(ifBias, metrics, fw);

            System.out.println("Min training Error:\t\t\t" + trainSNLF.min_Error);
            System.out.println("Min total training epochs:\t\t" + trainSNLF.min_Round);
            System.out.println("Total Round:\t\t" + trainSNLF.total_Round);
            System.out.println("Min total training time:\t\t" + trainSNLF.minTotalTime);
            System.out.println("Min average training time:\t\t" + trainSNLF.minTotalTime / trainSNLF.min_Round);
            System.out.println("Total training time:\t\t" + trainSNLF.total_Time);
            System.out.println("Average training time:\t\t" + trainSNLF.total_Time / trainSNLF.total_Round);

            fw.write("Min training Error:\t\t\t" + trainSNLF.min_Error + "\n");
            fw.write("Min total training epochs:\t\t" + trainSNLF.min_Round + "\n");
            fw.write("Total Round:\t\t" + trainSNLF.total_Round + "\n");
            fw.write("Min total training time:\t\t" + trainSNLF.minTotalTime + "\n");
            fw.write("Min average training time:\t\t" + trainSNLF.minTotalTime / trainSNLF.min_Round + "\n");
            fw.write("Total training time:\t\t" + trainSNLF.total_Time + "\n");
            fw.write("Average training time:\t\t" + trainSNLF.total_Time / trainSNLF.total_Round + "\n");
            fw.flush();
        }
        fw.close();
    }

    public void train(boolean ifBias, int metrics, FileWriter fw) throws IOException {

        double lastErr = 0;
        for (int round = 1; round <= maxRound; round++){

            double startTime = System.currentTimeMillis();

            // 每一轮开始时，将辅助矩阵元素置为0
            resetAuxArray();

            for(RTuple trainR: trainDataSet){

                //计算预测值
                double ratingHat = getPrediction(trainR.userID, trainR.itemID, ifBias);

                if(ifBias){
                    // 记录Bias上的学习增量
                    biasUp[trainR.userID] += trainR.rating;
                    biasDown[trainR.userID] += ratingHat;
                }

                // 记录隐特征矩阵上的学习增量
                for(int dim = 0; dim < featureDimension; dim++){
                    featureUp[trainR.userID][dim] += featureMatrix[trainR.itemID][dim] * trainR.rating;
                    featureDown[trainR.userID][dim] += featureMatrix[trainR.itemID][dim] * ratingHat;
                }
            }

            // 将所有的增量，根据结点的ID，加到相应的隐向量上
            for(int id = 1; id <= maxID; id++){

                if(ifBias){
                    // 更新Bias
                    biasDown[id] += bias[id] * RSetSize[id] * lambda;
                    if(biasDown[id] != 0)
                        bias[id] = bias[id] * (biasUp[id] / biasDown[id]);
                }

                // 更新FeatureMatrix
                for(int dim = 0; dim < featureDimension; dim++){
                    featureDown[id][dim] += featureMatrix[id][dim] * RSetSize[id] * lambda;
                    if(featureDown[id][dim] != 0)
                        featureMatrix[id][dim] = featureMatrix[id][dim] * (featureUp[id][dim] / featureDown[id][dim]);
                }
            }

            double endTime = System.currentTimeMillis();
            cacheMinTotalTime += endTime - startTime;
            total_Time += endTime - startTime;

            // 计算本轮训练结束后，在测试集上的误差
            double curErr;
            if (metrics == CommonRec.RMSE) {
                curErr = testRMSE(ifBias);
            } else {
                curErr = testMAE(ifBias);
            }
            fw.write(curErr + "\n");
            fw.flush();
            //System.out.println(curErr);

            total_Round += 1;
            if (min_Error > curErr) {
                min_Error = curErr;
                min_Round = round;
                minTotalTime += cacheMinTotalTime;
                cacheMinTotalTime = 0;
//                saveMinFeatureMatrix();
            }

            if (Math.abs(curErr - lastErr) > minGap)
                lastErr = curErr;
            else
                break;
        }

         outPutfeaturematrix();

    }

}
