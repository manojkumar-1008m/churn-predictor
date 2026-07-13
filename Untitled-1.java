import java.util.*;
public class Main{
    public static void main(String[]args){
        int arr[]={1,2,3,4,5};
        int target=7;
        for(int i=0;i<arr.length;i++){
            if(arr[i]==target){
                System.out.print("found");
                break;
            }else{
             System.out.print("not found");
            }
        }
    }
}