package com.JJLin.aiagent.entites;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("user_profile")
public class UserProfile {

    @TableId(value = "user_id", type = IdType.INPUT)
    private String userId;

    private Integer age;

    private String occupation;

    private String education;

    @TableField("income_range")
    private String incomeRange;

    private String gender;

    private String city;

    @TableField("marital_status")
    private String maritalStatus;

    private String notes;

    @TableField("created_at")
    private LocalDateTime createdAt;

    @TableField("updated_at")
    private LocalDateTime updatedAt;
}